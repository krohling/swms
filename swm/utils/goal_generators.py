import random
from PIL import Image
from abc import ABC
import numpy as np
from swm.utils.base_classes import BaseEnv, BaseGoalGenerator
from swm.semantic_world_model import SWMModel
import typing as tp
import mujoco


class LangTableBaseGoalGenerator(BaseGoalGenerator, ABC):
    def __init__(self, env: BaseEnv, model: SWMModel, answer_options: tp.List, kwargs: tp.Dict):
        super().__init__(env, model, answer_options, kwargs)
        self.reward_function_cls = kwargs["reward_function_cls"]
        block_combo = kwargs['block_combo']
        self.reward_function = self.reward_function_cls(
            goal_reward=1, block_combo=block_combo, rng=self.env._rng, delay_reward_steps=0, block_mode=self.env._block_mode
        )
        self.info = None
        self.step = 0
        self.block_combo = block_combo

    def get_frame(self):
        obs = self.env.compute_state()
        image = Image.fromarray(obs["rgb"])
        return image

    def get_instruction(self):
        return self.info.instruction
    
    def get_done(self):
        return self.reward_function.reward(self.env._compute_state(request_task_update=False))[0] > 0
    
    def reset_env(self, seed=None):
        from language_table.environments.rewards import task_info
        while True:
            ts = self.env.reset()
            info = self.reward_function.reset(
            self.env._compute_state(request_task_update=False),
                    # Define an instruction over just
                    # these blocks.
                    blocks_on_table=self.env._blocks_on_table)
            if info != task_info.FAILURE:
                break
        self.info = info
        self.reset_hook()
        return np.array(ts.observation["rgb"])
    
    def reset_hook(self):
        """Function called on environment reset after new info set"""
        pass
    
    def get_ckpt(self):
        return {"step": self.step}
    
    def load_from_ckpt(self, ckpt: tp.Dict):
        self.step = ckpt['step']

    
class PushBlocksTogetherGoal(LangTableBaseGoalGenerator):
    
    def get_questions(self):
        start_block = self.info.block1.replace("_", " ")
        end_block = self.info.block2.replace("_", " ")
        question = [{
                "text": f"Is the {start_block} touching the {end_block}?",
                "answer": self.answer_options[0],
                "weight": 0.8
            },
            {
                "text": f"Are the {start_block} and {end_block} closer together?",
                "answer": self.answer_options[0],
                "weight": 0.2
            },
        ]
        questions = [(q["text"], q["answer"], q["weight"]) for q in question]
        return questions
    
    def get_done(self):
        state = self.env._compute_state(request_task_update=False)
        first_block, _ = self.reward_function._get_pose_for_block(self.info.block1, state)
        second_block, _ = self.reward_function._get_pose_for_block(self.info.block2, state)
        dist = np.linalg.norm(np.array(first_block) - np.array(second_block))
        return dist < .065

class SeparateBlocksGoal(LangTableBaseGoalGenerator):
    def get_questions(self):
        center_block = self.info.block.replace("_", " ")
        avoid_blocks = [b.replace("_", " ") for b in self.info.avoid_blocks]

        question = [{
            "text": f"Is the robotic peg touching the {center_block}?",
            "answer": self.answer_options[0],
            "weight": .8
            },
        ]
        for avoid in avoid_blocks:
            question.append(
                {
                    "text": f"Is the {center_block} touching the {avoid}?",
                    "answer": self.answer_options[1],
                    "weight": 0.2 / len(avoid_blocks)
                })
        questions = [(q["text"], q["answer"], q["weight"]) for q in question]
        return questions


class PegToBlockGoal(LangTableBaseGoalGenerator):
    def get_questions(self):
        target_block = self.info.block_target
        question = [#{
            # "text": f"Is the robotic peg closer to the {target_block}?",
            # "answer": self.answer_options[0],
            # "weight": .5
            # },
            {
            "text": f"Is the robotic peg touching the {target_block}?",
            "answer": self.answer_options[0],
            "weight": 1.0
            }
        ]
        questions = [(q["text"], q["answer"], q["weight"]) for q in question]
        return questions



def get_lang_table_goal(reward_type:str, env, model, answer_options, kwargs) -> BaseGoalGenerator:
    from language_table.environments.rewards import point2block, separate_blocks, task_info, block2block, block2absolutelocation
    if reward_type == "peg_to_block":
        kwargs['reward_function_cls'] = point2block.PointToBlockReward
        return PegToBlockGoal(env, model, answer_options, kwargs)
    elif reward_type == "separate_blocks":
        kwargs['reward_function_cls'] =  separate_blocks.SeparateBlocksReward
        return SeparateBlocksGoal(env, model, answer_options, kwargs)
    elif reward_type == "block_to_block":
        kwargs['reward_function_cls'] = block2block.BlockToBlockReward
        return PushBlocksTogetherGoal(env, model, answer_options, kwargs)
    else:
        raise ValueError(f"Not supported reward type {reward_type}")


# ogbench


class OGBenchBaseGoalGenerator(BaseGoalGenerator, ABC):
    def __init__(self, env, model: SWMModel, answer_options: tp.List, kwargs: tp.Dict):
        super().__init__(env, model, answer_options, kwargs)
        self.block_to_number = {
            "red_cube": 0,
            "blue_cube": 1,
            "yellow_cube": 2,
            "green_cube": 3,
        }
        self.step = 0
        
        
    def get_frame(self):
        return Image.fromarray(self.env.unwrapped.get_pixel_observation())
    
    def reset_env(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.reset_hook()
        return obs
    
    def reset_hook(self):
        pass
    
    def get_ckpt(self):
        return {"step": self.step}
    
    def load_from_ckpt(self, ckpt: tp.Dict):
        self.step = ckpt['step']


class ReachBlockGoal(OGBenchBaseGoalGenerator):
    def __init__(self, env, model, answer_options, kwargs):
        super().__init__(env, model, answer_options, kwargs)
        self.block_goal = random.choice(list(self.block_to_number.keys()))
    
    def get_questions(self):
        """Return the questions that you should ask the model in the tuple list form"""
        question = [{
            "text": f"Is the robotic peg closer to the {self.block_goal.replace('_', ' ')}?",
            "answer": self.answer_options[0],
            "weight": 1# .2
            },
            {
            "text": f"Is the robotic peg touching the {self.block_goal.replace('_', ' ')}?",
            "answer": self.answer_options[0],
            "weight": 1# .8
            }
        ]
        questions = [(q["text"], q["answer"], q["weight"]) for q in question]
        return questions


    def get_instruction(self):
        return f"Reach the {self.block_goal.replace('_', ' ')}"
    
    def get_done(self):
        import mujoco
        unwrapped_env = self.env.unwrapped
        right_pad1_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/right_pad1')
        right_pad2_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/right_pad2')
        left_pad1_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/left_pad1')
        left_pad2_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/left_pad2')
        
        
        cube_id = unwrapped_env._cube_geom_ids_list[self.block_to_number[self.block_goal]][0]
        contacts = set()
        for contact_num in range(unwrapped_env._data.ncon):
            contact = unwrapped_env._data.contact[contact_num]
            if cube_id in contact.geom and (right_pad1_id in contact.geom or right_pad2_id in contact.geom or left_pad1_id in contact.geom or left_pad2_id in contact.geom):
                for geom in contact.geom:
                    contacts.add(geom)
        touching = len(contacts) > 0
        return touching


class StackBlocksGoal(OGBenchBaseGoalGenerator):
    def __init__(self, env, model, answer_options, kwargs):
        super().__init__(env, model, answer_options, kwargs)
        self.block_combo = kwargs["block_combo"]
        
    
    def reset_hook(self):
        self.step = 0
    
    def get_instruction(self):
        unwrapped_env = self.env.unwrapped

        bottom_cube_num = self.block_to_number[self.block_combo[1]]
        top_cube_num = self.block_to_number[self.block_combo[0]]
        
        bottom_cube = unwrapped_env._cube_geom_ids_list[bottom_cube_num][0]
        top_cube = unwrapped_env._cube_geom_ids_list[top_cube_num][0]

        return f"Stack the {top_cube} on top of the {bottom_cube}"
    
    def get_done(self):
        unwrapped_env = self.env.unwrapped
        bottom_cube_num = self.block_to_number[self.block_combo[1]]
        top_cube_num = self.block_to_number[self.block_combo[0]]
        
        bottom_cube = unwrapped_env._cube_geom_ids_list[bottom_cube_num][0]
        top_cube = unwrapped_env._cube_geom_ids_list[top_cube_num][0]
                
        contacts = unwrapped_env._data.contact.geom
        ontop = False
        if np.any(np.all(np.array([top_cube, bottom_cube]) == contacts, axis=1)) or np.any(np.all(np.array([bottom_cube, top_cube]) == contacts, axis=1)):
            # they are touching but check if one is on top of the other
            bottom_cube_pose, top_cube_pose = unwrapped_env._data.geom(bottom_cube).xpos, unwrapped_env._data.geom(top_cube).xpos
            if top_cube_pose[2] > bottom_cube_pose[2] and top_cube_pose[2] - bottom_cube_pose[2] > 0.015:
                ontop = True
            else:
                ontop = False
        else:
            ontop = False
        # check if the gripper is far away from the top and bottom cubes
        block_and_eef_poses = unwrapped_env.get_block_and_eef_poses()
        eef_pos = block_and_eef_poses['eef_pos']
        bottom_cube_pos = block_and_eef_poses[f'block_{bottom_cube_num}_pos']
        top_cube_pos = block_and_eef_poses[f'block_{top_cube_num}_pos']
        distance_to_bottom = np.linalg.norm(eef_pos - bottom_cube_pos)
        distance_to_top = np.linalg.norm(eef_pos - top_cube_pos)
        apart = distance_to_bottom > 0.05 and distance_to_top > 0.05
        
        right_pad1_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/right_pad1')
        right_pad2_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/right_pad2')
        left_pad1_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/left_pad1')
        left_pad2_id = mujoco.mj_name2id(unwrapped_env._model, mujoco.mjtObj.mjOBJ_GEOM, 'ur5e/robotiq/left_pad2')
        
        contacts = set()
        for contact_num in range(unwrapped_env._data.ncon):
            contact = unwrapped_env._data.contact[contact_num]
            if top_cube in contact.geom and (right_pad1_id in contact.geom or right_pad2_id in contact.geom or left_pad1_id in contact.geom or left_pad2_id in contact.geom):
                for geom in contact.geom:
                    contacts.add(geom)
        touching = len(contacts) > 0
        return ontop and not touching #apart

    
    def get_questions(self):
        first_block = self.block_combo[0].replace('_', ' ')
        second_block = self.block_combo[1].replace('_', ' ')
        if self.step == 0:
            # try to advance to step 1
            images = [self.get_frame()]
            scores = self.model.get_scores(images=images, actions = None, questions = [f"Is the robot grasping the {first_block}?"])
            yes_prob = scores[0].item()
            if yes_prob > .9: # .95
                self.step = 1
        if self.step == 0:
            question = [
                {
                "text": f"Is the robot grasping the {first_block}?",
                "answer": self.answer_options[0],
                "weight": 1.0
                }]
        else:
            question = [{
                "text": f"Is the {first_block} on top of the {second_block}?",
                "answer": self.answer_options[0],
                "weight": .6 # 1.0
                },
                {
                "text": f"Is the robot grasping the {first_block}?",
                "answer": self.answer_options[0],
                "weight": .4 # 1.0
                },
              ]
        questions = [(q["text"], q["answer"], q["weight"]) for q in question]
        return questions

def get_ogbench_goal(reward_type, env, model: SWMModel, answer_options, kwargs):
    if reward_type == 'reach_block':
        return ReachBlockGoal(env, model, answer_options, kwargs)
    elif reward_type == 'stack_blocks':
        return StackBlocksGoal(env, model, answer_options, kwargs)