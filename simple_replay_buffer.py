# simple_replay_buffer.py (最终修复版本)
import abc
import torch
import numpy as np
from replay_buffer import ReplayBuffer
import rlkit.torch.pytorch_util as ptu
import logging

# 导入安全转换工具
def safe_from_numpy(data, dtype=None):
    """安全地从各种类型转换为 PyTorch 张量"""
    try:
        if isinstance(data, torch.Tensor):
            if dtype is not None:
                return data.to(dtype)
            return data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        elif isinstance(data, (list, tuple)):
            if data and isinstance(data[0], np.ndarray):
                stacked = np.stack(data)
                tensor = torch.from_numpy(stacked)
            else:
                tensor = torch.tensor(data)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        else:
            tensor = torch.tensor(data)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
    except Exception as e:
        logging.error(f"张量转换失败: {str(e)}, 数据类型: {type(data)}")
        raise RuntimeError(f"无法转换数据为张量: {str(e)}")

class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, max_path_length, env_spec, auto_convert_to_tensor=True):
        self._max_size = max_size
        self.max_path_length = max_path_length
        self.env_spec = env_spec
        self.auto_convert_to_tensor = auto_convert_to_tensor
        self._data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "terminals": [],
            "labels": [],
        }
        self._current_size = 0

    def add_paths(self, paths):
        """添加路径数据到缓冲区 - 新增方法"""
        for path in paths:
            # 获取路径数据
            observations = path['observations']
            actions = path['actions']
            rewards = path['rewards']
            next_observations = path['next_observations']
            terminals = path['terminals']
            
            # 从env_infos中提取标签
            labels = []
            if 'env_infos' in path and path['env_infos']:
                for info in path['env_infos']:
                    if isinstance(info, dict):
                        if 'labels' in info:
                            labels.append(info['labels'])
                        elif 'true_label' in info:
                            labels.append(info['true_label'])
                        else:
                            labels.append(0)  # 默认值
                    else:
                        labels.append(0)
            else:
                # 如果没有env_infos，使用默认标签
                labels = [0] * len(observations)
            
            # 确保所有数组长度一致
            path_length = len(observations)
            if len(labels) != path_length:
                labels = labels[:path_length] + [0] * (path_length - len(labels))
            
            # 逐步添加每个样本
            for i in range(path_length):
                try:
                    # 确保数据类型正确
                    obs = observations[i] if isinstance(observations[i], np.ndarray) else np.array(observations[i], dtype=np.float32)
                    next_obs = next_observations[i] if isinstance(next_observations[i], np.ndarray) else np.array(next_observations[i], dtype=np.float32)
                    action = actions[i]
                    reward = rewards[i]
                    terminal = terminals[i]
                    label = labels[i]
                    
                    self.add_sample(
                        observation=obs,
                        action=action,
                        reward=reward,
                        next_observation=next_obs,
                        terminal=terminal,
                        labels=label
                    )
                except Exception as e:
                    logging.error(f"添加样本失败 (路径{len(paths)}, 步骤{i}): {str(e)}")
                    logging.error(f"  观测值类型: {type(observations[i])}, 形状: {getattr(observations[i], 'shape', 'N/A')}")
                    logging.error(f"  动作类型: {type(actions[i])}, 值: {actions[i]}")
                    logging.error(f"  奖励类型: {type(rewards[i])}, 值: {rewards[i]}")
                    continue

    def add_sample(self, observation, action, reward, next_observation, terminal, **kwargs):
        # ===== 关键修复：处理动作类型 =====
        # 转换动作为整数
        if isinstance(action, (np.ndarray, torch.Tensor)):
            # 如果是数组或张量，提取标量值
            if action.ndim > 0:
                action = action.squeeze()
            action = int(action.item() if hasattr(action, 'item') else action)
        elif not isinstance(action, int):
            # 尝试转换为整数
            try:
                action = int(action)
            except (TypeError, ValueError) as e:
                raise TypeError(f"无法将动作转换为整数: {type(action)}, 值: {action}") from e
        
        # ===== 关键修复：处理奖励类型（增强版）=====
        # 确保奖励是浮点数
        if isinstance(reward, (np.ndarray, torch.Tensor)):
            # 如果是数组或张量，先尝试squeeze去除多余维度
            if hasattr(reward, 'squeeze'):
                reward = reward.squeeze()
            
            # 检查是否为标量
            if hasattr(reward, 'ndim') and reward.ndim == 0:
                # 0维数组，直接转换
                reward = float(reward)
            elif hasattr(reward, 'shape') and reward.shape == ():
                # 空形状，直接转换
                reward = float(reward)
            elif hasattr(reward, 'numel') and reward.numel() == 1:
                # PyTorch张量，只有一个元素
                reward = float(reward.item())
            elif hasattr(reward, 'size') and reward.size == 1:
                # NumPy数组，只有一个元素
                reward = float(reward.item())
            else:
                # 尝试获取第一个元素
                try:
                    if hasattr(reward, '__getitem__'):
                        reward = float(reward[0])
                    else:
                        reward = float(reward)
                except Exception as e:
                    logging.error(f"奖励转换失败 - 类型: {type(reward)}, 形状: {getattr(reward, 'shape', 'unknown')}, 值: {reward}")
                    raise TypeError(f"无法将奖励转换为浮点数: {type(reward)}, 形状: {getattr(reward, 'shape', 'unknown')}") from e
        elif not isinstance(reward, (float, int)):
            # 尝试转换为浮点数
            try:
                reward = float(reward)
            except (TypeError, ValueError) as e:
                raise TypeError(f"无法将奖励转换为浮点数: {type(reward)}, 值: {reward}") from e
        else:
            # 确保是浮点数
            reward = float(reward)
        
        # ===== 关键修复：处理标签类型 =====
        labels = kwargs.get('labels', None)
        if labels is not None:
            if isinstance(labels, (np.ndarray, torch.Tensor)):
                if hasattr(labels, 'squeeze'):
                    labels = labels.squeeze()
                labels = int(labels.item() if hasattr(labels, 'item') else labels)
            elif not isinstance(labels, int):
                try:
                    labels = int(labels)
                except (TypeError, ValueError) as e:
                    raise TypeError(f"无法将标签转换为整数: {type(labels)}, 值: {labels}") from e
        
        # 添加类型验证
        if not isinstance(observation, np.ndarray):
            raise TypeError(f"观测值应为NumPy数组, 实际类型: {type(observation)}")
        if not isinstance(next_observation, np.ndarray):
            raise TypeError(f"下一观测值应为NumPy数组, 实际类型: {type(next_observation)}")
        if not isinstance(terminal, (bool, np.bool_)):
            # 尝试转换为布尔值
            try:
                terminal = bool(terminal)
            except Exception:
                raise TypeError(f"终止标志应为布尔值, 实际类型: {type(terminal)}")
        
        # 添加样本到缓冲区
        if self._current_size < self._max_size:
            self._data["observations"].append(observation)
            self._data["actions"].append(action)
            self._data["rewards"].append(reward)
            self._data["next_observations"].append(next_observation)
            self._data["terminals"].append(terminal)
            self._data["labels"].append(labels)
            self._current_size += 1
        else:
            # 覆盖旧数据
            idx = self._current_size % self._max_size
            self._data["observations"][idx] = observation
            self._data["actions"][idx] = action
            self._data["rewards"][idx] = reward
            self._data["next_observations"][idx] = next_observation
            self._data["terminals"][idx] = terminal
            self._data["labels"][idx] = labels
            self._current_size += 1

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self, **kwargs):
        return self._current_size

    def random_batch(self, batch_size):
        if self._current_size == 0:
            # 生成虚拟数据
            dummy_obs = np.zeros(
                (batch_size, self.env_spec.observation_space.shape[0]),
                dtype=np.float32
            )
            dummy_actions = np.zeros((batch_size,), dtype=np.int64)
            
            # 使用安全转换函数
            return {
                "observations": safe_from_numpy(dummy_obs, torch.float32).to(ptu.device),
                "actions": safe_from_numpy(dummy_actions, torch.long).to(ptu.device),
                "rewards": safe_from_numpy(np.zeros(batch_size, dtype=np.float32), torch.float32).to(ptu.device),
                "next_observations": safe_from_numpy(dummy_obs, torch.float32).to(ptu.device),
                "terminals": safe_from_numpy(np.zeros(batch_size, dtype=bool), torch.bool).to(ptu.device),
                "labels": safe_from_numpy(np.zeros(batch_size, dtype=np.int64), torch.long).to(ptu.device),
            }
        else:
            indices = np.random.randint(0, min(self._current_size, len(self._data["observations"])), size=batch_size)
            batch = {}
            
            # ============= 关键修复：使用安全转换函数 =============
            for key in self._data:
                try:
                    # 收集原始数据
                    raw_data = [self._data[key][i] for i in indices]
                    
                    # 如果启用自动转换，直接转换为张量
                    if self.auto_convert_to_tensor:
                        if key == "actions" or key == "labels":
                            # 整数类型
                            tensor_data = safe_from_numpy(raw_data, torch.long).to(ptu.device)
                        elif key == "rewards":
                            # 浮点数类型
                            tensor_data = safe_from_numpy(raw_data, torch.float32).to(ptu.device)
                        elif key == "terminals":
                            # 布尔类型
                            tensor_data = safe_from_numpy(raw_data, torch.bool).to(ptu.device)
                        else:
                            # 观测值：浮点数类型
                            tensor_data = safe_from_numpy(raw_data, torch.float32).to(ptu.device)
                        batch[key] = tensor_data
                    else:
                        # 保持为NumPy数组
                        if key in ["actions", "labels"]:
                            batch[key] = np.array(raw_data, dtype=np.int64)
                        elif key == "rewards":
                            batch[key] = np.array(raw_data, dtype=np.float32)
                        elif key == "terminals":
                            batch[key] = np.array(raw_data, dtype=bool)
                        else:
                            batch[key] = np.array(raw_data, dtype=np.float32)
                    
                except Exception as e:
                    logging.error(f"❌ 处理批次数据 {key} 时出错: {str(e)}")
                    raise
            
            return batch

    def get_diagnostics(self):
        return {"current_size": self._current_size}