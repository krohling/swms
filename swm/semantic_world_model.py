import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from swm.constants import ANSWER_OPTIONS
from swm.paligemma_wm import PaliGemmaWMForConditionalGeneration, PaliGemmaWMProcessor
from swm.utils.dataset import SWMRewardDataset


class SWMModel(ABC):
    @abstractmethod
    def __init__(self, checkpoint_path, processor_path, tokens, precision, device):
        '''Initialize the SWM model with the given parameters.'''
        pass

    @abstractmethod
    def get_scores(self, images, actions, questions) -> tp.Tuple[np.ndarray, ...]:
        '''Get the exact scores based on this model for the given images, actions, and questions.'''
        pass

    @abstractmethod
    def get_probabilistic_rewards_wm(self, action_seq, image, pred_horizon, questions, batch_size=64, action_skip=1) -> \
    tp.Tuple[np.ndarray, np.ndarray]:
        '''Get the rewards based on this model for the given action sequence, image, prediction horizon, and questions. Returns rewards and weighted rewards.'''
        pass



class SWMGradModel(SWMModel):
    # Same action ranking; sigmoid gradients vanish in saturated states, logit_margin's don't.
    OBJECTIVES = ("sigmoid", "logit_margin")

    def __init__(self, checkpoint_path, processor_path, tokens=ANSWER_OPTIONS,
                 precision=torch.float16, device="cuda", objective="sigmoid"):
        if objective not in self.OBJECTIVES:
            raise ValueError(f"objective must be one of {self.OBJECTIVES}, got {objective!r}")
        self.objective = objective
        self.processor = PaliGemmaWMProcessor.from_pretrained(processor_path)
        self.tokens = tokens
        self.token_to_id = {token: self.processor.tokenizer.encode(token) for token in tokens}
        self.model = None

        model = PaliGemmaWMForConditionalGeneration.from_pretrained(checkpoint_path, torch_dtype=precision).to(device)

        self.precision = precision
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

        self.device = device

    def get_scores(self, images, actions, questions, return_logits=False):
        if isinstance(questions, str):
            questions = [questions] * len(images)

        inputs = self.processor(text=questions, images=images, actions=actions,
                                return_tensors="pt", padding="longest",
                                tokenize_newline_separately=False).to(self.device, dtype=self.precision)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :].to(torch.float32)  # Get logits for the last token
        probs = torch.softmax(logits, dim=1)
        results = tuple(probs[:, self.token_to_id[token]].squeeze(1) for token in self.tokens)
        if return_logits:
            token_logits = tuple(logits[:, self.token_to_id[token]].squeeze(1) for token in self.tokens)
            return results, token_logits
        return results

    def get_probabilistic_rewards_wm(self, action_seq, image, pred_horizon, questions, batch_size=64, action_skip=1,
                                     gradient=False):
        weights = [x[2] for x in questions]
        with torch.no_grad() if not gradient else torch.enable_grad():
            rewards = np.zeros((len(questions), len(action_seq), pred_horizon), dtype=np.float32)
            rewards_with_grad_sum = 0
            data_loader = DataLoader(
                SWMRewardDataset(image=image, action_seq=action_seq, questions=questions, pred_horizon=pred_horizon,
                                 action_skip=action_skip),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=SWMRewardDataset.collate_fn,
                pin_memory=True
            )
            # Process batches
            for batch in tqdm(data_loader, desc="Processing batches", disable=True):
                # Get batch data
                batch_images = batch.pop('images')
                batch_actions = batch.pop('actions')
                batch_questions = batch.pop('questions')
                probs_tuple, logits_tuple = self.get_scores(batch_images, batch_actions, batch_questions,
                                                            return_logits=True)


                q_indices = batch['q_indices']
                a_indices = batch['a_indices']
                h_steps = batch['h_steps']
                desired_tokens = batch['desired_tokens']

                for idx in range(len(q_indices)):
                    q_idx = q_indices[idx]
                    a_idx = a_indices[idx]
                    h_step = h_steps[idx]
                    desired_token = desired_tokens[idx]
                    # Select correct probability based on desired token
                    if desired_token == self.tokens[0]:
                        prob = probs_tuple[0][idx]
                        margin = logits_tuple[0][idx] - logits_tuple[1][idx]
                    elif desired_token == self.tokens[1]:
                        prob = probs_tuple[1][idx]
                        margin = logits_tuple[1][idx] - logits_tuple[0][idx]
                    else:
                        raise ValueError(f"Invalid desired token: {desired_token}")
                    d_reward = prob.detach() if gradient else prob
                    # Reported rewards stay probabilities; objective only changes the differentiated term.
                    rewards[q_idx, a_idx, h_step - action_skip: h_step] = d_reward.cpu().numpy()
                    opt_term = prob if self.objective == "sigmoid" else margin
                    rewards_with_grad_sum += opt_term * weights[q_idx]
            # Apply weights to rewards
            weighted_rewards = rewards.copy()
            for i in range(len(weights)):
                weighted_rewards[i] *= weights[i]
        if gradient:
            return rewards, weighted_rewards, rewards_with_grad_sum
        return rewards, weighted_rewards
