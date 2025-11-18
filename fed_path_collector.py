# fed_path_collector.py (彻底修复版)
import torch
import numpy as np
import logging
from rlkit.samplers.data_collector import MdpPathCollector
from intrusion_detection_env import IntrusionDetectionEnv
from rlkit.samplers.rollout_functions import rollout

class FedPathCollector(MdpPathCollector):
    def __init__(
        self,
        policy,
        device,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        rollout_fn=None,
        save_env_in_snapshot=False
    ):
        # 提供默认 rollout 函数
        if rollout_fn is None:
            rollout_fn = self._custom_rollout
            
        super().__init__(
            env=None,
            policy=policy,
            max_num_epoch_paths_saved=max_num_epoch_paths_saved,
            render=render,
            render_kwargs=render_kwargs,
            rollout_fn=rollout_fn,
            save_env_in_snapshot=save_env_in_snapshot
        )
        self.device = device
        self.data_loader = None

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def _validate_observation(self, obs, step_info=""):
        """验证观测值格式"""
        if not isinstance(obs, np.ndarray):
            logging.error(f"{step_info}观测值不是NumPy数组: 类型={type(obs)}")
            raise TypeError(f"观测值应为NumPy数组，实际类型: {type(obs)}")
        
        if obs.ndim == 0:
            logging.error(f"{step_info}观测值是0维标量: {obs}")
            logging.error(f"这表明环境返回了标量而非特征向量")
            raise ValueError(f"{step_info}观测值不能是0维标量")
        
        if obs.ndim != 1:
            logging.error(f"{step_info}观测值维度异常: {obs.ndim}维, 形状={obs.shape}")
            raise ValueError(f"{step_info}观测值应为1维数组")
        
        return True

    def _custom_rollout(
        self,
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
    ):
        """
        自定义rollout函数，彻底修复观测值处理问题
        """
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        
        # 环境重置
        try:
            o = env.reset()
            self._validate_observation(o, "环境重置后")
            logging.debug(f"Rollout开始: 初始观测值形状={o.shape}, 类型={o.dtype}")
        except Exception as e:
            logging.error(f"环境重置失败: {str(e)}")
            raise
        
        agent.reset()
        path_length = 0
        
        if render:
            env.render(**render_kwargs)
            
        while path_length < max_path_length:
            # 再次验证当前观测值
            try:
                self._validate_observation(o, f"步骤{path_length}")
            except Exception as e:
                logging.error(f"步骤{path_length}观测值验证失败: {str(e)}")
                logging.error(f"当前观测值: 类型={type(o)}, 形状={getattr(o, 'shape', 'N/A')}, 值={o}")
                raise
            
            # 获取动作 - 直接传递观测值，不做任何修改
            try:
                a, agent_info = agent.get_action(o)
                logging.debug(f"步骤{path_length}: 动作={a}, 观测值形状={o.shape}")
            except Exception as e:
                logging.error(f"步骤{path_length}获取动作失败: {str(e)}")
                logging.error(f"传递给策略的观测值: 类型={type(o)}, 形状={o.shape}")
                logging.error(f"观测值内容: {o}")
                raise
            
            # 执行动作
            try:
                next_o, r, d, env_info = env.step(a)
                self._validate_observation(next_o, f"步骤{path_length}环境返回")
            except Exception as e:
                logging.error(f"步骤{path_length}环境执行失败: {str(e)}")
                raise
            
            # 存储轨迹数据
            observations.append(o.copy())  # 使用copy确保数据独立
            rewards.append(float(r))
            terminals.append(bool(d))
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            
            if d:
                break
                
            # 关键点：直接赋值，不做任何处理
            o = next_o
            
            if render:
                env.render(**render_kwargs)
        
        # 转换为数组
        if observations:
            # 验证所有观测值形状一致性
            first_shape = observations[0].shape
            for i, obs in enumerate(observations):
                if obs.shape != first_shape:
                    logging.error(f"观测值{i}形状不一致: 期望{first_shape}, 实际{obs.shape}")
                    raise ValueError(f"轨迹中观测值形状不一致")
            
            observations = np.array(observations)
            logging.debug(f"观测值数组最终形状: {observations.shape}")
        else:
            expected_shape = env.observation_space.shape
            observations = np.zeros((0, *expected_shape), dtype=np.float32)
        
        # 构建next_observations数组
        next_observations = np.zeros_like(observations)
        if len(observations) > 1:
            next_observations[:-1] = observations[1:]
        
        # 处理最后一个next_observation
        if path_length > 0:
            if 'next_o' in locals() and next_o is not None:
                if next_o.shape == observations[0].shape:
                    next_observations[-1] = next_o
                else:
                    next_observations[-1] = observations[-1]
            else:
                next_observations[-1] = observations[-1] if len(observations) > 0 else np.zeros(env.observation_space.shape)
        
        # 确保数据类型正确
        rewards = np.array(rewards, dtype=np.float32)
        terminals = np.array(terminals, dtype=bool)
        
        logging.debug(f"Rollout完成: 观测值形状={observations.shape}, 路径长度={path_length}")
        
        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            agent_infos=agent_infos,
            env_infos=env_infos,
        )

    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths=False):
        paths = []
        num_steps_collected = 0
        
        while num_steps_collected < num_steps:
            # 确保data_loader已设置
            if self.data_loader is None:
                raise ValueError("DataLoader未设置，请先调用set_data_loader()")
            
            # 采样数据并严格验证
            try:
                features, labels = self.data_loader.sample_batch()
                
                # 严格类型检查
                if not isinstance(features, np.ndarray):
                    raise TypeError(f"特征应为NumPy数组，实际类型: {type(features)}")
                if not isinstance(labels, np.ndarray):
                    raise TypeError(f"标签应为NumPy数组，实际类型: {type(labels)}")
                
                # 严格维度检查
                expected_batch_size = self.data_loader.batch_size
                expected_feature_dim = self.data_loader.feature_dim
                
                if features.shape != (expected_batch_size, expected_feature_dim):
                    logging.error(f"特征形状错误: 期望({expected_batch_size}, {expected_feature_dim}), 实际{features.shape}")
                    raise ValueError(f"特征数据形状错误")
                
                if labels.shape != (expected_batch_size,):
                    logging.error(f"标签形状错误: 期望({expected_batch_size},), 实际{labels.shape}")
                    raise ValueError(f"标签数据形状错误")
                
                # 类型转换
                if features.dtype != np.float32:
                    features = features.astype(np.float32)
                if labels.dtype != np.int64:
                    labels = labels.astype(np.int64)
                
                logging.debug(f"数据采样验证通过: 特征{features.shape}, 标签{labels.shape}")
                
            except Exception as e:
                logging.error(f"数据采样或验证失败: {str(e)}")
                raise
            
            # 创建环境
            try:
                env = IntrusionDetectionEnv(
                    data_loader=self.data_loader,
                    max_steps=max_path_length
                )
                
                # 直接设置环境数据，不通过任何中间处理
                env.current_batch = features
                env.current_labels = labels
                env.current_step = 0
                env.current_idx = 0
                
                logging.debug(f"环境创建成功，批次数据已设置")
                
            except Exception as e:
                logging.error(f"环境创建失败: {str(e)}")
                raise
            
            # 执行rollout
            try:
                path = self._rollout_fn(
                    env,
                    self._policy,
                    max_path_length=min(max_path_length, len(features))
                )
                
                logging.debug(f"Rollout执行成功: 观测值形状={path['observations'].shape}")
                
            except Exception as e:
                logging.error(f"Rollout执行失败: {str(e)}")
                raise
            
            # 后处理路径数据
            if 'actions' in path:
                actions = path['actions']
                if isinstance(actions, np.ndarray):
                    if actions.ndim > 1:
                        actions = actions.squeeze()
                    path['actions'] = actions.astype(np.int64)
            
            # 从env_infos中提取标签
            if 'env_infos' in path and path['env_infos']:
                labels_list = []
                for info in path['env_infos']:
                    if 'labels' in info:
                        labels_list.append(info['labels'])
                    elif 'true_label' in info:
                        labels_list.append(info['true_label'])
                    else:
                        labels_list.append(0)
                path['labels'] = np.array(labels_list, dtype=np.int64)
            
            paths.append(path)
            num_steps_collected += len(path['observations'])
            
            logging.debug(f"路径收集完成: 总步数={num_steps_collected}/{num_steps}")
            
            if num_steps_collected >= num_steps:
                break
        
        return paths

    def get_snapshot(self):
        return {'policy': self._policy}