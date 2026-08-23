from typing import Any, Tuple
import numpy as np
from PIL import Image
from swm.utils.base_classes import BaseEnv
import gymnasium

class LangTableEnv(BaseEnv):
    def step(self, action):
        """Take an environment step, and return just the rgb observation as a numpy unit8 array."""
        ts = self.env.step(action)
        current_frame = np.array(ts.observation["rgb"])
        return current_frame

    def get_state(self):
        return self.env.get_pybullet_state()

    def project_actions_to_camera_frame(self, actions: np.ndarray) -> np.ndarray:
        actions = np.cumsum(actions, axis=0)
        actions = self.get_eef_pose() + actions
        pix_x, pix_y = self.env.get_camera_pix_coords(actions)
        return pix_x, pix_y

    def get_eef_pose(self):
        peg_pose = self.env.get_block_states()["peg"]
        return peg_pose

    @property
    def scale_factor(self) -> float:
        return 0.03

    def get_frame(self):
        obs = self.env.compute_state()
        image = Image.fromarray(obs["rgb"])
        return image


def get_lang_table_env(kwargs, seed=54):
    # Language-table deps are heavy (TF) and only needed for LT runs; import lazily
    # so ogbench-only environments (e.g. local macOS) don't require them.
    from language_table.environments.blocks import LanguageTableBlockVariants
    from language_table.environments.lang_table_data_generation import LanguageTableDataGeneration
    from language_table.environments.rewards.noop import NoOpReward
    from tf_agents.environments import gym_wrapper
    import tensorflow as tf
    # fix for gpu error in language table
    tf.config.experimental.set_visible_devices([], "GPU")

    ood = kwargs['ood']
    block_mode = LanguageTableBlockVariants.BLOCK_8 if not ood else LanguageTableBlockVariants.NOVEL_8
    env = LanguageTableDataGeneration(
        block_mode=block_mode,
        reward_factory=NoOpReward,
        seed=seed,
        block_combo=kwargs['block_combo'],
    )

    # changed to false so it does not automatically reset
    env = gym_wrapper.GymWrapper(env, auto_reset=False)

    env.reset()
    env = LangTableEnv(env)
    return env



class OGBenchEnv(BaseEnv):
    def step(self, action) -> np.ndarray:
        """Take an environment step, and return just the rgb observation as a numpy unit8 array."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs

    def get_state(self) -> Any:
        """Return the current env state so that you can log it"""
        return self.env.unwrapped.get_state()
    
    def set_state(self, state: Any) -> None:
        """Set the environment state to a given state."""
        self.env.unwrapped.set_state_from_dict(state)

    def get_original_env(self) -> Any:
        """Return the original environment, so that you can access the original env methods for goal generation, etc."""
        return self.env
    
    def get_frame(self):
        return Image.fromarray(self.env.unwrapped.get_pixel_observation())


    def project_actions_to_camera_frame(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points to the camera frame of the environment. return the pixel coordinates (x, y) of the points."""
        peg_pose = self.get_eef_pose()
        actions = actions[..., :3]  # Only take the first 3 dimensions (x, y, z)
        # scale the actions by the range
        actions = actions * self.env.unwrapped.action_high[:3]
        actions = np.cumsum(actions, axis=0)  # Cumulative sum to get the trajectory
        actions = peg_pose + actions  # Add the end-effector pose to the actions

        projection_matrix = self.env.unwrapped.get_camera_matrices()
        assert actions.shape[1] == 3, f"Expected xyz actions after slicing, got shape {actions.shape}"
        actions = np.concatenate([actions, np.ones((actions.shape[0], 1))], axis=1)

        clip_coords = projection_matrix @ actions.T

        # Perform perspective division (divide by w component)
        # clip_coords is now 4xN, we need the first 3 components

        # Extract 2D screen coordinates (assuming NDC to screen space conversion)
        # Convert from normalized device coordinates [-1,1] to screen coordinates [0,width/height]
        z_vals = clip_coords[2]
        pixel_x = np.floor(clip_coords[0] / z_vals)
        pixel_y = np.floor(clip_coords[1] / z_vals)  # pixel+x = np.floor((x + 1) * width / 2)
        return pixel_x, pixel_y

    def get_eef_pose(self) -> np.ndarray:
        """Get the end-effector pose in the world frame."""
        pose = self.env.unwrapped.get_block_and_eef_poses()['eef_pos']
        return pose

    @property
    def scale_factor(self) -> float:
        """Return the scale factor for action sampling i.e. magnitude of the actions."""
        return 1.0


def get_ogbench_env(kwargs, seed=None) -> BaseEnv:
    # from ogbench.manipspace.envs.cube_env_for_data_collection import CubeEnvForDataCollection
    env = gymnasium.make(
        'visual-cube-quadruple-v0',
        terminate_at_goal=False,
        visualize_info=False,
        mode='data_collection',
        max_episode_steps=1000, 
        width=224,
        height=224,
        control_timestep=0.1,
        stack_goal=None, # don't want the environment to have the goal
        ood=kwargs['ood'],
    )
    env = OGBenchEnv(env)
    return env
