from abc import ABC, abstractmethod
from typing import Any, Tuple
from PIL import Image
import numpy as np
import typing as tp


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

class BaseEnv(ABC):
    """This is the interface I interact with the environments with."""

    def __init__(self, env):
        self.env = env

    @property
    @abstractmethod
    def scale_factor(self) -> float:
        """Return the scale factor of the environment, used for scaling the actions."""

    @abstractmethod
    def step(self, action) -> np.ndarray:
        """Take an environment step, and return just the rgb observation as a numpy unit8 array."""

    @abstractmethod
    def get_state(self) -> Any:
        """Return the current env state so that you can log it"""

    def get_original_env(self) -> Any:
        """Return the original environment, so that you can access the original env methods for goal generation, etc."""
        return self.env

    @abstractmethod
    def project_actions_to_camera_frame(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points to the camera frame of the environment. return the pixel coordinates (x, y) of the points."""

    @abstractmethod
    def get_eef_pose(self) -> np.ndarray:
        """Get the end-effector pose in the world frame."""

    @abstractmethod
    def get_frame(self) -> Image.Image:
        """Return the env current frame as a PIL image"""
        pass


class BaseGoalGenerator(ABC):
    def __init__(self, env: BaseEnv, model: SWMModel, answer_options: tp.List, kwargs: tp.Dict):
        self.env = env.get_original_env()
        self.answer_options = answer_options
        self.model: SWMModel = model
        self.step = 0

    @abstractmethod
    def get_questions(self):
        """Return the questions that you should ask the model in the tuple list form"""
        pass
    @abstractmethod
    def get_frame(self) -> Image.Image:
        """Return the env current frame as a PIL image"""
        pass
    
    @abstractmethod
    def get_instruction(self) -> str:
        """Return the instruction for the current task"""
        pass
    
    @abstractmethod
    def get_done(self) -> bool:
        """return if the episode or goal has succeeded"""
        pass
    
    @abstractmethod
    def reset_env(self, seed=None) -> np.ndarray:
        """Reset the environment and return the initial observation"""
        pass
    
    def get_ckpt(self) -> tp.Dict:
        """Return the checkpoint of the goal generator, used for saving and loading"""
        return {}
    
    def load_from_ckpt(self, ckpt: tp.Dict):
        """Load the goal generator from a checkpoint"""
        pass
    
