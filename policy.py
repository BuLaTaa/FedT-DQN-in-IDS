# policy.py (修复策略接口兼容性)
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from rlkit.torch.networks import Mlp
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
import rlkit.torch.pytorch_util as ptu
import logging

# 设置日志级别
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')

class ClassificationPolicy(Mlp, TorchStochasticPolicy):
    """
    入侵检测专用策略网络
    输入：观测特征（网络流量特征）
    输出：离散动作概率分布（0: 正常，1: 异常）
    """

    def __init__(self, hidden_sizes, obs_dim, action_dim=2, init_w=1e-3, hidden_activation=nn.ReLU, output_activation=None, temperature=1.0, layer_normalization=False, **kwargs):
        assert action_dim == 2, "入侵检测应为二元分类"
        
        super().__init__(hidden_sizes=hidden_sizes, input_size=obs_dim, output_size=action_dim, init_w=init_w, hidden_activation=hidden_activation(), output_activation=output_activation, **kwargs)
        self.device = ptu.device
        self.to(self.device)
        
        # 添加层标准化（可选）
        self.layer_norm = nn.LayerNorm(hidden_sizes[-1]) if layer_normalization else None
        
        # 初始化分类参数
        self.temperature = temperature
        self._init_fc_layers()
        
        # 添加初始化验证
        try:
            self._validate_initialization()
        except RuntimeError as e:
            logging.error(f"策略网络初始化失败: {str(e)}")
            raise
    
    def _validate_initialization(self):
        """验证网络是否合理初始化"""
        self.to(self.device)
        test_input = torch.randn(1, self.input_size).to(self.device)
        with torch.no_grad():
            dist = self.forward(test_input)
            probs = dist.probs
            
            if torch.all(probs == probs[0,0]):
                raise RuntimeError("策略网络初始化失败：输出恒定值")
            if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                raise RuntimeError("策略网络输出包含NaN/Inf")
    
    def _init_fc_layers(self):
        """分类专用层初始化"""
        for layer in self.fcs:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.1)
        nn.init.normal_(self.last_fc.weight, mean=0, std=1e-2)

    def forward(self, obs):
        """
        输入：obs - 观测特征 (batch_size, obs_dim)
        输出：分类分布 (batch_size, action_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        elif obs.dim() != 2:
            raise ValueError(f"输入维度异常: {obs.shape}")
        
        if obs.shape[1] != self.input_size:
            raise ValueError(f"输入特征维度错误: 预期 {self.input_size}, 实际 {obs.shape[1]}")
        
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
            if self.layer_norm:
                h = self.layer_norm(h)
                
        logits = self.last_fc(h) / self.temperature
        dist = Categorical(logits=logits)
        
        return dist

    def logprob(self, actions, dist):
        """计算动作的日志概率"""
        return dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def _preprocess_observation(self, obs):
        """预处理观测值，确保格式正确"""
        expected_dim = self.input_size
        
        # 精简日志信息
        logging.debug(f"预处理输入: 类型={type(obs)}, 维度={getattr(obs, 'ndim', 'N/A')}, 形状={getattr(obs, 'shape', 'N/A')}")
        
        # ============= 关键修复：优先处理张量类型 =============
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy().astype(np.float32)
        
        if isinstance(obs, np.floating):
            obs = np.full(expected_dim, obs, dtype=np.float32)
        elif isinstance(obs, float):
            obs = np.full(expected_dim, obs, dtype=np.float32)
        
        if isinstance(obs, tuple):
            obs = obs[0]  # 只保留第一个元素
        
        if not isinstance(obs, (np.ndarray, torch.Tensor)):
            if hasattr(obs, '__array__') or hasattr(obs, '__iter__'):
                try:
                    obs = np.array(obs, dtype=np.float32)
                except Exception as e:
                    logging.error(f"无法转换为数组: {str(e)}")
                    raise TypeError(f"观测数据无法转换为数组: {type(obs)}")
            else:
                raise TypeError(f"不支持的观测数据类型: {type(obs)}")
        
        # 确保是NumPy数组
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        
        if isinstance(obs, np.ndarray):
            if obs.ndim == 0:
                obs = np.full(expected_dim, obs, dtype=np.float32)
            elif obs.ndim == 1:
                if obs.shape[0] != expected_dim:
                    logging.error(f"1维观测值特征数错误: 期望{expected_dim}, 实际{obs.shape[0]}")
                    raise ValueError(f"特征维度不匹配: 期望{expected_dim}, 实际{obs.shape[0]}")
            elif obs.ndim == 2:
                if obs.shape == (1, expected_dim):
                    obs = obs.squeeze(0)
                elif obs.shape == (expected_dim, 1):
                    obs = obs.squeeze(1)
                else:
                    raise ValueError(f"2维观测值形状错误: {obs.shape}")
            
            if obs.dtype != np.float32:
                obs = obs.astype(np.float32)
            
            # ============= 关键修复：确保转换为张量时在正确设备上 =============
            obs = torch.from_numpy(obs).float().to(self.device)
        
        if obs.dim() == 1 and obs.shape[0] == expected_dim:
            obs = obs.unsqueeze(0)
        elif obs.dim() == 2 and obs.shape[1] == expected_dim:
            pass
        else:
            raise ValueError(f"预处理后张量形状仍然错误: {obs.shape}")
        
        logging.debug(f"预处理完成: 输出形状={obs.shape}, 设备={obs.device}")
        return obs

    def get_action(self, obs, deterministic=False):
        """
        获取分类动作 - 修复返回格式兼容性
        
        返回格式：
        - action: int 或 numpy array
        - info: dict (包含 log_prob)
        """
        logging.debug(f"Policy get_action (ENTRY): obs details -> 类型={type(obs)}")
        try:
            processed_obs = self._preprocess_observation(obs)
        
            with torch.no_grad():
                dist = self.forward(processed_obs)
                action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
                log_prob = dist.log_prob(action)
            
                action_value = action.item() if action.dim() > 0 else int(action)
                log_prob_value = log_prob.item() if log_prob.dim() > 0 else float(log_prob)
                
                logging.debug(f"策略网络输出: 动作={action_value}, 日志概率={log_prob_value}")
            
            # ============= 修复：返回兼容格式 =============
            # 返回 (action, agent_info) 格式，兼容标准rlkit接口
            agent_info = {
                'log_prob': log_prob_value,
                'mean': dist.probs.cpu().numpy() if dist.probs.dim() > 0 else float(dist.probs),
                'pre_tanh_value': action_value
            }
            
            return action_value, agent_info
        
        except Exception as e:
            logging.error(f"策略网络get_action失败: {str(e)}")
            raise

    # ============= 新增：添加标准rlkit策略接口方法 =============
    def get_actions(self, obs, deterministic=False):
        """批量获取动作 - 标准rlkit接口"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        actions = []
        agent_infos = []
        
        for i in range(batch_size):
            action, agent_info = self.get_action(obs[i], deterministic)
            actions.append(action)
            agent_infos.append(agent_info)
        
        return np.array(actions), agent_infos

    def reset(self):
        """重置策略状态 - 标准rlkit接口"""
        pass

    def train(self, mode=True):
        """设置训练模式"""
        super().train(mode)
        return self

    def eval(self):
        """设置评估模式"""
        return self.train(False)

class MultiHeadClassificationPolicy(ClassificationPolicy):
    """联邦Former适配版：支持Transformer特征融合"""
    def __init__(self, transformer_layers=2, **kwargs):
        super().__init__(**kwargs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_sizes[-1], nhead=4, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
    def forward(self, obs):
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
            if self.layer_norm:
                h = self.layer_norm(h)
                
        h = self.transformer_encoder(h.unsqueeze(0)).squeeze(0)

        logits = self.last_fc(h) / self.temperature
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("Logits 包含 NaN 或 Inf，请检查输入数据和网络参数！")
        logits = torch.clamp(logits, min=-50, max=50)
        dist = Categorical(logits=logits)
        return dist