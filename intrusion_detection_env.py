# intrusion_detection_env.py 
import numpy as np
import gym
from gym import spaces
import logging
import torch 


logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')

class IntrusionDetectionEnv(gym.Env):
    """
    
    """
    
    def __init__(self, data_loader, max_steps=1000):
        super().__init__()
        self.data_loader = data_loader
        self.max_steps = max_steps
        

        self.action_space = spaces.Discrete(2)
        

        feature_dim_val = int(self.data_loader.feature_dim)
        if feature_dim_val <= 0:
            raise ValueError(f"Feature dimension must be positive, got {feature_dim_val}")
            
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim_val,),
            dtype=np.float32
        )
        

        self.current_step = 0
        self.current_batch = None
        self.current_labels = None
        self.current_idx = 0
        
   

    def _get_observation(self, idx):
  
        if self.current_batch is None:
            logging.error("当前批次数据为空，请先调用reset()")
            raise RuntimeError("当前批次数据为空，请先调用reset()")

    
        if self.current_batch.ndim != 2:
            raise ValueError(f"current_batch 必须是二维数组，实际维度: {self.current_batch.ndim}")
        
        feature_dim = self.observation_space.shape[0]
        if self.current_batch.shape[1] != feature_dim:
            raise ValueError(f"特征维度不匹配: 期望 {feature_dim}, 实际 {self.current_batch.shape[1]}")
        

        obs_arr = self.current_batch[idx, :].astype(np.float32)
        
    
        obs_arr = obs_arr.flatten()
        if obs_arr.shape[0] != feature_dim:
            logging.error("观测值维度错误")
            raise ValueError(f"观测值维度错误: 期望({feature_dim},)维, 实际{obs_arr.shape}维")
        
        logging.debug(f"返回 obs 形状 {obs_arr.shape}")
        return obs_arr
    
    def reset(self):
        self.current_step = 0
        self.current_idx = 0


        self.current_batch, self.current_labels = self.data_loader.sample_batch()
        if self.current_batch is None:
            logging.error("尝试加载数据时当前批次为空")
            raise RuntimeError("Failed to load data in reset.")
            
 
        if self.current_batch.ndim == 1:
          
            self.current_batch = self.current_batch.reshape(1, -1)
        elif self.current_batch.ndim != 2:
            raise ValueError(f"current_batch 维度不支持: {self.current_batch.ndim}")

        observation = self._get_observation(self.current_idx)
    
        if not isinstance(observation, np.ndarray) or observation.ndim != 1 or observation.shape[0] != self.observation_space.shape[0]:
            logging.error("重置时观测值验证失败")
            raise TypeError("重置: 观测值不是正确形状的1D NumPy数组")
    
        return observation
    
    def step(self, action):
     
        if isinstance(action, torch.Tensor):
            action_val = action.item()
        elif isinstance(action, np.ndarray):
            action_val = action.item() if action.size == 1 else action[0]
        elif isinstance(action, (list, tuple)):
            action_val = action[0]
        elif isinstance(action, (int, float)):
            action_val = int(action)
        else:
            raise ValueError(f"动作必须为数字或可迭代对象，实际类型: {type(action)}")
        
        action_val = int(action_val)  # 确保为整数
        if action_val not in [0, 1]:
            raise ValueError(f"动作值必须为0或1，实际值: {action_val}")


        true_label_val = int(self.current_labels[self.current_idx])
        

        if action_val == true_label_val:
            reward = 2.0 if true_label_val == 1 else 1.0
        else:
          
            if true_label_val == 1 and action_val == 0:
                reward = -3.0
            else:
                reward = -1.0
        
        self.current_step += 1
        self.current_idx = (self.current_idx + 1) % len(self.current_batch)
        
        done = (self.current_step >= self.max_steps) or (self.current_idx == 0 and self.current_step > 0)

    
        if done:
            next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_observation = self._get_observation(self.current_idx)

        info = {
            'true_label': true_label_val,
            'predicted_label': action_val,
            'correct': (action_val == true_label_val),
            'labels': true_label_val
        }
        
        logging.debug(f"环境步骤完成: action={action_val}, reward={reward}, done={done}")
        return next_observation, float(reward), bool(done), info
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    @property
    def spec(self):
    
        class EnvSpec:
            def __init__(self, observation_space, action_space, max_episode_steps):
                self.observation_space = observation_space
                self.action_space = action_space
                self.max_episode_steps = max_episode_steps
                
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            max_episode_steps=self.max_steps
        )
