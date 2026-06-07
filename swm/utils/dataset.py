import os
import pickle as pkl

import numpy as np
import torch
from torch.utils.data import Dataset


class SWMRewardDataset(Dataset):
    def __init__(self, image, action_seq, questions, pred_horizon, action_skip):
        self.image = image
        self.action_seq = action_seq
        if isinstance(self.action_seq, np.ndarray):
            self.action_seq = torch.from_numpy(self.action_seq)
        self.questions = questions
        self.pred_horizon = pred_horizon
        self.action_skip = action_skip

        # Build index of all tasks (q_idx, a_idx, h_step)
        self.tasks = []
        for q_idx in range(len(questions)):
            for a_idx in range(len(action_seq)):
                for h_step in range(action_skip, pred_horizon + action_skip, action_skip):
                    self.tasks.append((q_idx, a_idx, h_step))

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        q_idx, a_idx, h_step = self.tasks[idx]
        question_str, desired_token, _ = self.questions[q_idx]

        # Get action sequence up to current horizon step
        action = self.action_seq[a_idx, :h_step]

        return {
            'image': self.image,
            'action': action,
            'question': question_str,
            'q_idx': q_idx,
            'a_idx': a_idx,
            'h_step': h_step,
            'desired_token': desired_token
        }

    @staticmethod
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        actions = [item['action'] for item in batch]
        questions = [item['question'] for item in batch]
        q_indices = [item['q_idx'] for item in batch]
        a_indices = [item['a_idx'] for item in batch]
        h_steps = [item['h_step'] for item in batch]
        desired_tokens = [item['desired_token'] for item in batch]
        return {
            'images': images,
            'actions': actions,
            'questions': questions,
            'q_indices': q_indices,
            'a_indices': a_indices,
            'h_steps': h_steps,
            'desired_tokens': desired_tokens
        }


class DiffusionDataset(Dataset):
    def __init__(self, data_folder_path, horizon=10, obs_horizon=1, pad_length=12):
        self.horizon = horizon
        trajectories = []
        for file in os.listdir(data_folder_path):
            if file.endswith(".pkl"):
                data_file = os.path.join(data_folder_path, file)
                data_file = pkl.load(open(data_file, "rb"))
                images = np.array(data_file["frames"], dtype=np.uint8)  # Keep uint8 until normalization
                actions = np.array(data_file["actions"], dtype=np.float32)  # Use float32 early
                trajectories.append((images, actions))
        self.images, self.actions, self.idxs = [], [], []
        current_total_frames_offset = 0
        current_total_actions_offset = 0
        # start_stop is a list of tuples where it is (frame, ac_start, ac_stop)
        for trajectory in trajectories:
            t_im, t_ac = trajectory
            # pad images with the last frame if necessary
            if pad_length > 0:
                padding = np.tile(t_im[-1:], (pad_length, 1, 1, 1))
                t_im = np.concatenate([t_im, padding], axis=0)
                padding = np.zeros((pad_length, t_ac.shape[1]), dtype=np.float32)
                t_ac = np.concatenate([t_ac, padding], axis=0)

            self.images.append(t_im)
            self.actions.append(t_ac)
            act_idxs = [(i, i + self.horizon) for i in
                        range(current_total_actions_offset, current_total_actions_offset + len(t_ac) - self.horizon)]
            frame_idxs = [tuple(current_total_frames_offset + max(0, i - j) for j in
                                reversed(range(obs_horizon))) for i in range(len(act_idxs))]
            self.idxs.extend(list(zip(frame_idxs, act_idxs)))
            current_total_frames_offset += len(t_im)
            current_total_actions_offset += len(t_ac)

        self.images = np.concatenate(self.images, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.images = torch.tensor(self.images).permute(0, 3, 1, 2)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)

        self.stats = {"action": get_data_stats(self.actions)}
        self.actions = normalize_data(self.actions, self.stats["action"])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        frame_tuple, (ac_start, ac_stop) = self.idxs[idx]
        images = [self.images[i].to(torch.float32) / 255.0 for i in frame_tuple]
        images = torch.stack(images, dim=0)
        return {
            "image": images.clone(),
            "action": self.actions[ac_start:ac_stop].clone()
        }

# normalize data
def get_data_stats(data):
    stats = {
        'min': data.min(axis=0).values,
        'max': data.max(axis=0).values
    }
    return stats

def normalize_data(data, stats):
    # normalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
