import os
import pickle
import random
import shutil
import time
from typing import Dict

import numpy as np
import torch

from swm.constants import ANSWER_OPTIONS
from swm.utils.visualizations import get_heat_map, save_video
from swm.utils.envs import get_lang_table_env, get_ogbench_env
from swm.semantic_world_model import SWMModel, SWMGradModel
from swm.utils.goal_generators import get_lang_table_goal, BaseGoalGenerator, get_ogbench_goal
from swm.planning_algos import PlanningConfig, get_plan
from swm.diffusion_policy.policy import DiffusionPolicy


def eval(
        seed: int,
        reward_type: str,
        env_type: str,

        # file paths
        output_dir: str,
        ckpt_path: str,
        processor_path: str,
        model: SWMModel = None,
        diffusion_path: str = None,
        device: str = "cuda",

        diffusion: bool = False,
        mppi: bool = False,
        gradient: bool = False,
        expert_diffusion: bool = False,
        
        # Model parameters
        precision=torch.bfloat16,
        action_skip: int = 5,
        model_batch_size=32,

        # env parameters
        reward_kwargs: Dict = {},
        action_dim: int = 2,

        # general planning params
        num_steps: int = 30,
        num_actions_executed: int = 5,
        pred_horizon: int = 80,
        num_samples: int = 64,
        num_planning_iters: int = 10,

        # gradient planning params
        gradient_lr: float = 0.01,
        gradient_clipping_value: float = 1.0,

        # mppi params
        mppi_temperature: float = 1.5,

        # visualization parameters
        intermediate_hm: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if os.path.exists(output_dir):
        # check for a reward.pkl file
        if os.path.exists(os.path.join(output_dir, "reward.pkl")):
            with open(os.path.join(output_dir, "reward.pkl"), "rb") as f:
                reward, time_taken = pickle.load(f)
            return reward, time_taken
        else: 
            shutil.rmtree(output_dir) # remove the directory if it exists but no reward.pkl file
            os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)
    
    if env_type == "lang_table":
        env = get_lang_table_env(kwargs=reward_kwargs, seed=seed)
    elif env_type == "ogbench":
        env = get_ogbench_env(kwargs=reward_kwargs, seed=seed)
    else:
        raise ValueError(f"Invalid env_type: {env_type}")

    
    pln_cfg = PlanningConfig(
        action_skip=action_skip,
        pred_horizon=pred_horizon,
        num_samples=num_samples,
        mppi_temperature=mppi_temperature,
        n_planning_itrs=num_planning_iters,
        action_dim=action_dim,
        max_action_value=env.scale_factor,
        diffusion=diffusion,
        mppi=mppi,
        gradient=gradient,
        gradient_lr=gradient_lr,
        gradient_clipping_value=gradient_clipping_value,
        batch_size=model_batch_size
    )
    
    if not expert_diffusion:
        if model is None:
            model = SWMGradModel(checkpoint_path=ckpt_path, processor_path=processor_path, tokens=ANSWER_OPTIONS,
                                precision=precision, device=device)
    else: 
        model = None
    
    if env_type == "lang_table":
        goal_generator: BaseGoalGenerator = get_lang_table_goal(reward_type, env, model, ANSWER_OPTIONS, reward_kwargs)
    elif env_type == "ogbench":
        goal_generator: BaseGoalGenerator = get_ogbench_goal(reward_type, env, model, ANSWER_OPTIONS, reward_kwargs)

    current_frame = goal_generator.reset_env(seed=seed)


    video_to_save = [current_frame]

    diffusion_model = None
    if diffusion or expert_diffusion:
        diffusion_model = DiffusionPolicy.load(diffusion_path, device=device)
        diffusion_model.add_obs(current_frame)
    
    sim_states = []
    actions = []
    start_time = time.time()
    sim_states.append(env.get_state())

    with open(os.path.join(output_dir, "goal.txt"), "w") as f:
        goal = goal_generator.get_instruction()
        f.write(goal + "\n")

    for i in range(num_steps):
        if expert_diffusion:
            best_action_seq = diffusion_model.get_action()
            action_seq = best_action_seq[np.newaxis]
            hm = get_heat_map(env, current_frame, action_seq, np.zeros((1, 1, 1)))
        else:
            questions = goal_generator.get_questions()
            with open(os.path.join(output_dir, "goal.txt"), "a") as f:
                f.write(f"{i}: \n")
                f.write(str(questions))

            best_action_seq, action_seq, hm = get_plan(
                env=env,
                current_obs_arr=current_frame,
                model=model,
                diffusion_model=diffusion_model,
                questions=questions,
                pln_cfg=pln_cfg,
                intermediate_hm=intermediate_hm,
            )

        if not isinstance(hm, list):
            hm = [hm]
        for j in range(len(hm)):
            hm[j].save(os.path.join(output_dir, f"heat_map_{i}_{j}.png"))

        for action in best_action_seq[:num_actions_executed]:
            actions.append(action)
            current_frame = env.step(action)
            sim_states.append(env.get_state())

            if diffusion or expert_diffusion:
                diffusion_model.add_obs(current_frame)

            video_to_save.append(current_frame)

            reward = goal_generator.get_done()
            if reward:
                break
     
        if reward:
            break
    end_time = time.time()
    save_video(video_to_save, file_path=os.path.join(output_dir, "video.mp4"), fps=10)
    # dump the pickle file of the reward for checkpointing
    with open(os.path.join(output_dir, "reward.pkl"), 'wb') as f:
        pickle.dump((reward, (end_time - start_time) / 60), f)
    return reward, (end_time - start_time) / 60

