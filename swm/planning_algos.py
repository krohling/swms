import torch
import numpy as np
from PIL import Image
from PIL import Image
import numpy as np
from functools import partial
from dataclasses import dataclass
from swm.utils.visualizations import get_heat_map


@dataclass
class PlanningConfig:
    """Dataclass to hold constants used for planning.
    """
    action_skip: int
    pred_horizon: int 
    num_samples: int 
    mppi_temperature: int 
    n_planning_itrs: int
    action_dim: int 
    max_action_value: float
    diffusion: bool = False
    mppi: bool = False    
    gradient: bool = False
    gradient_lr: float = 0.01
    gradient_clipping_value: float = 1.0
    batch_size: int = 64

def get_plan(
    env,
    current_obs_arr,
    model,
    diffusion_model,
    questions,
    pln_cfg: PlanningConfig,
    intermediate_hm=False,
):
    current_pil = Image.fromarray(current_obs_arr)
    reward_fn = partial(
        model.get_probabilistic_rewards_wm,
        image=current_pil,
        pred_horizon=pln_cfg.pred_horizon,
        questions=questions,
        batch_size=pln_cfg.batch_size,
        action_skip=pln_cfg.action_skip,
    )

    if pln_cfg.mppi:
        mppi_results = plan_model_mppi(
            diffusion_model=diffusion_model,
            get_rewards_fn=reward_fn,
            pln_cfg=pln_cfg,
            ret_intermediate=intermediate_hm,
        )
        if not intermediate_hm:
            action_seq, rewards, weighted_rewards = mppi_results
        else:
            action_seq, rewards, weighted_rewards, action_hist, weighted_rewards_hist = mppi_results
    elif pln_cfg.gradient:
        gradient_results = plan_model_gradient(
            diffusion_model=diffusion_model,
            get_rewards_fn=reward_fn,
            pln_cfg=pln_cfg,
            ret_intermediate=intermediate_hm,
        )
        if not intermediate_hm:
            action_seq, rewards, weighted_rewards = gradient_results
        else:
            action_seq, rewards, weighted_rewards, action_hist, weighted_rewards_hist = gradient_results
    else:
        action_seq = sample_initial_actions(diffusion_model=diffusion_model, pln_cfg=pln_cfg)
        rewards, weighted_rewards = reward_fn(action_seq)

        # Sum rewards and pick best action sequence
    reward = weighted_rewards.sum(axis=2).sum(axis=0)
    
    best_idx = np.argmax(reward)
    best_action_seq = action_seq[best_idx]

    # Generate heat map
    if intermediate_hm and (pln_cfg.mppi or pln_cfg.gradient):
        hm = []
        for i in range(len(action_hist)):
            hm.append(
                get_heat_map(
                    env, current_obs_arr, action_hist[i], weighted_rewards_hist[i]
                )
            )
    else:
        hm = get_heat_map(env, current_obs_arr, action_seq, weighted_rewards)
    return best_action_seq, action_seq, hm



def sample_initial_actions(diffusion_model, pln_cfg: PlanningConfig):
    if diffusion_model is not None:
        random_actions = torch.from_numpy(diffusion_model.sample_trajs(pln_cfg.num_samples))
    else:
        random_actions = torch.FloatTensor(pln_cfg.num_samples, pln_cfg.pred_horizon, pln_cfg.action_dim).uniform_(-1, 1) * pln_cfg.max_action_value
    return random_actions


def plan_model_mppi(
                    diffusion_model,
                    get_rewards_fn,
                    pln_cfg: PlanningConfig,
                    ret_intermediate=False,
                    ):
    # Sampling random actions in the range of the action space
    actions_hist = []
    weighted_rewards_hist = []
    random_actions = sample_initial_actions(diffusion_model=diffusion_model, pln_cfg=pln_cfg)
    random_actions = torch.clamp(random_actions, -pln_cfg.max_action_value, pln_cfg.max_action_value)
    # Rolling forward through the model for horizon steps

    rewards, weighted_rewards = get_rewards_fn(random_actions, action_skip=pln_cfg.action_skip)
    # Take first action from best trajectory
    if ret_intermediate:
        actions_hist.append(random_actions.cpu().detach().numpy())
        weighted_rewards_hist.append(weighted_rewards.copy())

    all_returns = weighted_rewards.sum(axis=2).sum(axis=0)

    # Run through a few iterations of MPPI
    for iter in range(pln_cfg.n_planning_itrs):
        # Weight trajectories by exponential of returns
        weights = torch.softmax(torch.from_numpy(all_returns) / pln_cfg.mppi_temperature, dim=0)[:, torch.newaxis, torch.newaxis]
        means = torch.sum(weights * random_actions, dim=0)
        stds = torch.sqrt(torch.sum(weights * (random_actions - means) ** 2, dim=0)) + 1e-9 # add in epsilon to the std distribution
        action_dist = torch.distributions.Normal(means, stds)
        # clamp these to -.2 to .2
        random_actions = action_dist.sample(sample_shape=(pln_cfg.num_samples, ))
        random_actions = torch.clamp(random_actions, -pln_cfg.max_action_value, pln_cfg.max_action_value)

        rewards, weighted_rewards = get_rewards_fn(random_actions, action_skip=pln_cfg.action_skip)

        if ret_intermediate:
            actions_hist.append(random_actions.cpu().detach().numpy())
            weighted_rewards_hist.append(weighted_rewards.copy())
        all_returns = weighted_rewards.sum(axis=2).sum(axis=0)
    
    if ret_intermediate:
        return random_actions, rewards, weighted_rewards, actions_hist, weighted_rewards_hist
    
    return random_actions, rewards, weighted_rewards


def plan_model_gradient(
                    diffusion_model,
                    get_rewards_fn,
                    pln_cfg: PlanningConfig,
                    ret_intermediate=False,
                    action_samp=None
                    ):
    if action_samp is None:
        action_samp = sample_initial_actions(diffusion_model=diffusion_model, pln_cfg=pln_cfg)
    original = action_samp.clone().numpy()
    action_samp.requires_grad = True
    action_hist, weighted_rewards_hist = [], []
    action_optimizer = torch.optim.SGD([action_samp], lr=pln_cfg.gradient_lr)
    for i in range(pln_cfg.n_planning_itrs):
        action_optimizer.zero_grad()

        rewards, weighted_rewards, rewards_with_grad_sum = get_rewards_fn(action_samp, action_skip=pln_cfg.action_skip, gradient=True)
        if ret_intermediate:
            action_hist.append(action_samp.cpu().detach().numpy().copy())
            weighted_rewards_hist.append(weighted_rewards.copy())

        loss = -rewards_with_grad_sum # We want to maximize rewards, so we minimize the negative
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_([action_samp], max_norm=pln_cfg.gradient_clipping_value)
        action_optimizer.step()

        # Have to clamp like this otherwise breaks optimizer
        with torch.no_grad():
            action_samp.clamp_(-pln_cfg.max_action_value, pln_cfg.max_action_value)


    action_samp = action_samp.detach().numpy()
    
    if ret_intermediate:
        return action_samp, rewards, weighted_rewards, action_hist, weighted_rewards_hist
    return action_samp, rewards, weighted_rewards
