from typing import Iterable
import torch.nn.functional as F
import torch
import torch.nn as nn
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import rlkit.torch.pytorch_util as ptu
import logging
import numpy as np
from sklearn.metrics import roc_auc_score  # 新增：导入AUC计算函数

class SACClassifierTrainer(TorchTrainer):
    def __init__(
        self,
        client_id,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        policy_lr=3e-4,
        qf_lr=3e-4,
        discount=0.99,
        soft_target_tau=0.005,
        cls_weights=None  # 新增：动态类别权重
    ):
        super().__init__()
        self.device = policy.device  # 关键修复：从 policy 中获取设备
        # ========== 关键修改：使用传入的类别权重 ==========
        if cls_weights is None:
            cls_weights = torch.tensor([1.0, 1.0], device=self.device)
        self.cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)
        # ==============================================
        self.client_id = client_id
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.current_stats = {}
        # 优化器
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.qf1_optim = torch.optim.Adam(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optim = torch.optim.Adam(self.qf2.parameters(), lr=qf_lr)

        # 损失函数
        self.q_criterion = nn.MSELoss()
        
    @property
    def networks(self) -> Iterable[nn.Module]:
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def _ensure_tensor(self, data, dtype=None, device=None):
        """确保数据是正确类型的张量 - 修复版本"""
        if device is None:
            device = self.device
            
        # ============= 关键修复：正确处理所有输入类型 =============
        if isinstance(data, torch.Tensor):
            # 如果已经是张量，直接使用
            tensor = data.to(device)
        elif isinstance(data, np.ndarray):
            # 如果是NumPy数组，转换为张量
            tensor = torch.from_numpy(data).to(device)
        elif isinstance(data, (list, tuple)):
            # 如果是列表或元组，先转换为NumPy数组再转换为张量
            tensor = torch.tensor(data, device=device)
        else:
            # 其他类型，直接转换为张量
            tensor = torch.tensor(data, device=device)
        
        # 应用数据类型转换
        if dtype is not None:
            tensor = tensor.to(dtype)
        
        return tensor

    def compute_loss(self, batch):
        # ============= 修复：安全的调试日志记录 =============
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("=== 训练器接收到的批次数据类型 ===")
            for key, value in batch.items():
                try:
                    shape_str = str(value.shape) if hasattr(value, 'shape') else 'N/A'
                    dtype_str = str(value.dtype) if hasattr(value, 'dtype') else 'N/A'
                    logging.debug(f"{key}: {type(value).__name__} shape={shape_str} dtype={dtype_str}")
                except Exception as e:
                    logging.debug(f"{key}: {type(value).__name__} (记录失败: {e})")
        
        # ============= 关键修复：确保所有批次数据类型正确 =============
        # 添加设备一致性检查
        batch_device = None
        for key, value in batch.items():
            if hasattr(value, 'device'):
                if batch_device is None:
                    batch_device = value.device
                elif batch_device != value.device:
                    logging.warning(f"设备不一致: {key} 在 {value.device}, 期望在 {batch_device}")
        
        if batch_device != self.device:
            logging.warning(f"批次设备 {batch_device} 与训练器设备 {self.device} 不匹配")
        
        # 安全的张量转换，确保设备一致
        obs = self._ensure_tensor(batch['observations'], dtype=torch.float32, device=batch_device)
        acts = self._ensure_tensor(batch['actions'], dtype=torch.long, device=batch_device).squeeze()
        next_obs = self._ensure_tensor(batch['next_observations'], dtype=torch.float32, device=batch_device)
        rews = self._ensure_tensor(batch['rewards'], dtype=torch.float32, device=batch_device).squeeze()
        dones = self._ensure_tensor(batch['terminals'], dtype=torch.float32, device=batch_device).squeeze()
        
        # 处理标签数据
        if 'labels' in batch:
            labels = self._ensure_tensor(batch['labels'], dtype=torch.long).squeeze()
        else:
            # 如果没有标签，尝试从env_infos中提取
            labels = []
            if 'env_infos' in batch:
                for info in batch['env_infos']:
                    if isinstance(info, dict) and 'labels' in info:
                        labels.append(info['labels'])
                    elif isinstance(info, dict) and 'true_label' in info:
                        labels.append(info['true_label'])
                    else:
                        labels.append(0)  # 默认值
            else:
                labels = torch.zeros_like(acts)  # 默认全为0
            labels = self._ensure_tensor(labels, dtype=torch.long).squeeze()

        # 策略网络前向传播
        try:
            dist = self.policy(obs)
            # 调试：检查分布对象
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Policy分布类型: {type(dist)}")
                logging.debug(f"Logits形状: {dist.logits.shape}, 类型: {dist.logits.dtype}")
                logging.debug(f"Probs形状: {dist.probs.shape}, 类型: {dist.probs.dtype}")
            
            sampled_acts = dist.sample()  # 采样的动作（用于策略更新）
            log_probs = dist.log_prob(sampled_acts)
            
            # 调试：检查采样结果
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"采样动作形状: {sampled_acts.shape}, 类型: {sampled_acts.dtype}")
                logging.debug(f"日志概率形状: {log_probs.shape}, 类型: {log_probs.dtype}")
            
            # 计算预测结果
            preds = torch.argmax(dist.probs, dim=1)
            
            # 调试：检查预测结果
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"预测结果形状: {preds.shape}, 类型: {preds.dtype}")
                
        except Exception as e:
            logging.error(f"策略网络前向传播失败: {str(e)}")
            logging.error(f"观测值形状: {obs.shape}, 类型: {obs.dtype}, 设备: {obs.device}")
            raise
    
        # 统计混淆矩阵
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
    
        # ========== 新增：计算AUC指标 ==========
        try:
            # 获取正类（类别1）的预测概率
            y_scores = dist.probs[:, 1].detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            
            # 检查是否有两个类别
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_scores)
                auc_tensor = torch.tensor(auc, device=self.device, dtype=torch.float32)
            else:
                # 如果只有一个类别，AUC无法计算，设为0.5
                auc_tensor = torch.tensor(0.5, device=self.device, dtype=torch.float32)
                logging.warning(f"客户端{self.client_id}: 批次中只有一个类别，AUC设为0.5")
        except Exception as e:
            logging.error(f"AUC计算失败: {str(e)}")
            auc_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        # =======================================

        # 添加类别权重（假设类别1是少数类，权重设为10）
        cls_loss = F.cross_entropy(
            dist.logits, 
            labels, 
           weight=torch.tensor([1.0, 10.0], dtype=torch.float32, device=self.device)  # 确保权重类型和设备正确
       )

        # Q 函数计算（使用预测动作 acts）
        act_onehot = torch.nn.functional.one_hot(acts.long(), 2).float()
        q1_pred = self.qf1(torch.cat([obs, act_onehot], dim=1)).view(-1)  # 从 [128,1] 压缩为 [128]
        q2_pred = self.qf2(torch.cat([obs, act_onehot], dim=1)).view(-1)
        
        with torch.no_grad():
            # 使用策略网络预测下一个动作
            next_dist = self.policy(next_obs)
            next_acts = next_dist.sample().detach()
            # ============= 关键修复：确保动作为整数类型 =============
            next_acts = next_acts.long()  # 确保为整数类型
            next_onehot = torch.nn.functional.one_hot(next_acts, 2).float()
            
            # 计算目标Q值
            q1_next = self.target_qf1(torch.cat([next_obs, next_onehot], 1))
            q2_next = self.target_qf2(torch.cat([next_obs, next_onehot], 1))
            target_q = torch.min(q1_next, q2_next).squeeze()
            q_target = rews + (1 - dones) * self.discount * target_q
        
        # 确保Q目标有合理范围
        q_target = torch.clamp(q_target, -10.0, 10.0)

        # Q 损失
        q1_loss = self.q_criterion(q1_pred, q_target)
        q2_loss = self.q_criterion(q2_pred, q_target)

        # 策略损失（结合分类损失）
        policy_loss = -torch.mean(q1_pred) + 0.5 * cls_loss
        accuracy = (torch.argmax(dist.probs, 1) == labels).float().mean()
        
        # ========== 修复精确率、召回率、F1计算 ==========
        # 安全的零值检查和张量创建
        tp_val = tp.item()
        fp_val = fp.item()  
        fn_val = fn.item()
        
        if tp_val + fp_val == 0:
            precision = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            precision = tp / (tp + fp + 1e-8)
            
        if tp_val + fn_val == 0:
            recall = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        else:
            recall = tp / (tp + fn + 1e-8)
            
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # ========== 更新：包含AUC的损失字典 ==========
        losses = {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'policy_loss': policy_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_tensor  # 新增AUC
        }
        
        # ========== 更新：包含AUC的统计信息 ==========
        self.current_stats = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'auc': auc_tensor.item()  # 新增AUC
        }  # 保存统计信息
        
        return losses

    def train(self, batch):
        """
        重写基类的train方法，处理张量和NumPy数组输入
        """
        try:
            # 调试输出
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("=== SACClassifierTrainer.train() 被调用 ===")
                for key, value in batch.items():
                    logging.debug(f"  {key}: {type(value)}")
            
            # 检查是否所有主要数据都是张量
            main_keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
            all_tensors = all(
                isinstance(batch.get(key), torch.Tensor) 
                for key in main_keys 
                if key in batch
            )
            
            if all_tensors:
                # 如果已经是张量，直接调用train_from_torch
                logging.debug("数据已为张量，直接调用 train_from_torch")
                return self.train_from_torch(batch)
            else:
                # 如果是NumPy数组，转换为张量
                logging.debug("转换NumPy数组为张量")
                tensor_batch = {}
                for key, value in batch.items():
                    if isinstance(value, np.ndarray):
                        if key in ['actions', 'labels']:
                            tensor_batch[key] = torch.from_numpy(value).long().to(self.device)
                        elif key in ['rewards']:
                            tensor_batch[key] = torch.from_numpy(value).float().to(self.device)
                        elif key in ['terminals']:
                            tensor_batch[key] = torch.from_numpy(value).bool().to(self.device)
                        else:
                            tensor_batch[key] = torch.from_numpy(value).float().to(self.device)
                    else:
                        tensor_batch[key] = value
                return self.train_from_torch(tensor_batch)
                
        except Exception as e:
            logging.error(f"SACClassifierTrainer.train() 失败: {str(e)}")
            logging.error(f"输入批次类型: {[(k, type(v)) for k, v in batch.items()]}")
            raise

    def train_from_torch(self, batch):
        losses = self.compute_loss(batch)
        
        # 更新策略网络
        self.policy_optim.zero_grad()
        losses['policy_loss'].backward(retain_graph=True)
        self.policy_optim.step()

        # ----------------- 更新 Q 网络 -----------------
        # 清空 Q 网络梯度
        self.qf1_optim.zero_grad()
        self.qf2_optim.zero_grad()
        # 合并 Q 网络损失（避免多次反向传播）
        total_q_loss = losses['q1_loss'] + losses['q2_loss']
        # 反向传播 Q 网络损失（不保留计算图）
        total_q_loss.backward()
        # 执行梯度更新
        self.qf1_optim.step()
        self.qf2_optim.step()

        # ----------------- 更新目标网络 -----------------
        self._update_target_networks()

        return losses
    
    def get_stats(self):
        """返回当前统计信息（供联邦聚合使用）"""
        return self.current_stats  # 新增方法
        
    def _update_target_networks(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)