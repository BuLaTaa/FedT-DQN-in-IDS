# fed_algorithm.py (修复版 - 包含AUC指标)
import logging
import numpy as np
import torch
from collections import OrderedDict
from rlkit.core import Logger
import rlkit.torch.pytorch_util as ptu
import copy

class FedAlgorithm:
    def __init__(self, algorithms, num_epochs, fedFormer=False, patience=5):
        self.algorithms = algorithms
        self.num_epochs = num_epochs
        self.fedFormer = fedFormer
        self.patience = patience
        self.logger = Logger()
        
        # ========== 更新：增强的指标跟踪 - 包含AUC指标 ==========
        self.metrics = {
            'global': {
                'accuracy': [], 'q1_loss': [], 'q2_loss': [], 'policy_loss': [],
                'precision': [], 'recall': [], 'f1': [], 'auc': []  # 新增AUC
            },
            'clients': {
                i: {
                    'accuracy': [], 'q1_loss': [], 'q2_loss': [], 'policy_loss': [],
                    'precision': [], 'recall': [], 'f1': [], 'auc': []  # 新增AUC
                } 
                for i in range(len(algorithms))
            }
        }
        # ======================================================
        
        # 早停相关状态
        self.best_global_f1 = -np.inf
        self.best_models = {}
        self.epochs_without_improvement = 0
        self.final_epoch = 0

    def train(self):
        if not self.algorithms:
            logging.error("客户端算法列表为空！")
            return
        
        logging.info(f"开始联邦训练，共 {self.num_epochs} 轮")
        for epoch in range(self.num_epochs):
            self.final_epoch = epoch + 1
            logging.info(f"=== 第 {epoch+1}/{self.num_epochs} 轮 ===")
            
            # ===== 1. 客户端本地训练 =====
            client_metrics = []
            for client_id, algo in enumerate(self.algorithms):
                try:
                    # 保存聚合前的模型作为参考
                    pre_policy = copy.deepcopy(algo.trainer.policy.state_dict())
                    
                    # 执行本地训练
                    algo.step(epoch)
                    
                    # 计算本地训练变化量
                    post_policy = algo.trainer.policy.state_dict()
                    delta = self._calc_model_delta(pre_policy, post_policy)
                    logging.debug(f"客户端 {client_id} 参数变化: {delta:.4f}")
                    
                    # ===== 修复：收集完整指标（包含AUC）=====
                    stats = algo.trainer.get_stats()
                    accuracy = stats.get('accuracy', 0.0)
                    q1_loss = stats.get('q1_loss', 0.0)
                    q2_loss = stats.get('q2_loss', 0.0)
                    policy_loss = stats.get('policy_loss', 0.0)
                    precision = stats.get('precision', 0.0)
                    recall = stats.get('recall', 0.0)
                    f1 = stats.get('f1', 0.0)
                    auc = stats.get('auc', 0.0)  # 新增AUC获取
                    
                    # 记录客户端指标（包含AUC）
                    self.metrics['clients'][client_id]['accuracy'].append(accuracy)
                    self.metrics['clients'][client_id]['q1_loss'].append(q1_loss)
                    self.metrics['clients'][client_id]['q2_loss'].append(q2_loss)
                    self.metrics['clients'][client_id]['policy_loss'].append(policy_loss)
                    self.metrics['clients'][client_id]['precision'].append(precision)
                    self.metrics['clients'][client_id]['recall'].append(recall)
                    self.metrics['clients'][client_id]['f1'].append(f1)
                    self.metrics['clients'][client_id]['auc'].append(auc)  # 新增AUC记录
                    
                    # ===== 修复：传递完整指标给全局计算（包含AUC）=====
                    client_metrics.append({
                        'accuracy': accuracy,
                        'q1_loss': q1_loss,
                        'q2_loss': q2_loss,
                        'policy_loss': policy_loss,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc  # 新增AUC传递
                    })
                    
                    logging.info(
                        f"[Client {client_id}] Epoch {epoch+1} "
                        f"Accuracy: {accuracy:.4f}, Q1 Loss: {q1_loss:.4f}, Q2 Loss: {q2_loss:.4f}, "
                        f"Policy Loss: {policy_loss:.4f}, Precision: {precision:.4f}, "
                        f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"  # 新增AUC显示
                    )
                except Exception as e:
                    logging.error(f"客户端 {client_id} 训练失败: {str(e)}")
            
            # ===== 2. 联邦聚合 =====
            if self.fedFormer:
                self._fuse_transformers()
            else:
                self._fedavg()
            
            # ===== 3. 修复：计算完整全局指标（包含AUC）=====
            global_accuracy = np.mean([m['accuracy'] for m in client_metrics])
            global_q1_loss = np.mean([m['q1_loss'] for m in client_metrics])
            global_q2_loss = np.mean([m['q2_loss'] for m in client_metrics])
            global_policy_loss = np.mean([m['policy_loss'] for m in client_metrics])
            global_precision = np.mean([m['precision'] for m in client_metrics])
            global_recall = np.mean([m['recall'] for m in client_metrics])
            global_f1 = np.mean([m['f1'] for m in client_metrics])
            global_auc = np.mean([m['auc'] for m in client_metrics])  # 新增全局AUC计算
            
            # ===== 修复：记录完整全局指标（包含AUC）=====
            self.metrics['global']['accuracy'].append(global_accuracy)
            self.metrics['global']['q1_loss'].append(global_q1_loss)
            self.metrics['global']['q2_loss'].append(global_q2_loss)
            self.metrics['global']['policy_loss'].append(global_policy_loss)
            self.metrics['global']['precision'].append(global_precision)
            self.metrics['global']['recall'].append(global_recall)
            self.metrics['global']['f1'].append(global_f1)
            self.metrics['global']['auc'].append(global_auc)  # 新增全局AUC记录
            
            logging.info(
                f"[Global] Epoch {epoch+1} "
                f"Accuracy: {global_accuracy:.4f}, Q1 Loss: {global_q1_loss:.4f}, "
                f"Q2 Loss: {global_q2_loss:.4f}, Policy Loss: {global_policy_loss:.4f}, "
                f"Precision: {global_precision:.4f}, Recall: {global_recall:.4f}, "
                f"F1: {global_f1:.4f}, AUC: {global_auc:.4f}"  # 新增全局AUC显示
            )
            
            # ===== 4. 早停逻辑（基于F1分数） =====
            if global_f1 > self.best_global_f1:
                self.best_global_f1 = global_f1
                self.epochs_without_improvement = 0
                
                # ========== 更新：保存最佳模型（包含AUC信息）==========
                self.best_models = {
                    'epoch': epoch,
                    'global_f1': global_f1,
                    'global_precision': global_precision,
                    'global_auc': global_auc,  # 新增：保存AUC
                    'global_q1_loss': global_q1_loss,
                    'client_models': [
                        {
                            'policy': copy.deepcopy(algo.trainer.policy.state_dict()),
                            'qf1': copy.deepcopy(algo.trainer.qf1.state_dict()),
                            'qf2': copy.deepcopy(algo.trainer.qf2.state_dict())
                        } for algo in self.algorithms
                    ]
                }
                logging.info(f"新的最佳模型 - F1: {global_f1:.4f}, Precision: {global_precision:.4f}, "
                           f"AUC: {global_auc:.4f}, Q1 Loss: {global_q1_loss:.4f}")  # 新增AUC显示
                # =========================================================
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logging.info(f"早停：连续 {self.patience} 轮全局F1未提升，终止训练")
                    break
        
        # ===== 5. 训练结束处理 =====
        self._finalize_training()
        
    def _fedavg(self):
        """改进的联邦平均，减少随机扰动"""
        policy_params = []
        qf1_params = []
        qf2_params = []
        
        for algo in self.algorithms:
            if hasattr(algo.trainer, 'policy'):
                policy_params.append(algo.trainer.policy.state_dict())
                qf1_params.append(algo.trainer.qf1.state_dict())
                qf2_params.append(algo.trainer.qf2.state_dict())
    
        # 对参数进行联邦平均（移除随机噪声）
        global_policy = self._average_params(policy_params, add_noise=False)
        global_qf1 = self._average_params(qf1_params, add_noise=False)
        global_qf2 = self._average_params(qf2_params, add_noise=False)
    
        # 分发全局参数
        for algo in self.algorithms:
            if hasattr(algo.trainer, 'policy'):
                algo.trainer.policy.load_state_dict(global_policy)
                algo.trainer.qf1.load_state_dict(global_qf1)
                algo.trainer.qf2.load_state_dict(global_qf2)

    def _average_params(self, params_list, add_noise=False):
        """参数平均，可选是否添加噪声"""
        averaged_params = OrderedDict()
        for key in params_list[0].keys():
            stacked = torch.stack([p[key] for p in params_list])
            mean_val = torch.mean(stacked, dim=0)
            
            # 仅在前几轮添加少量噪声
            if add_noise and self.final_epoch < 10:
                noise = torch.randn_like(mean_val) * 0.005
                mean_val += noise
                
            averaged_params[key] = mean_val
        return averaged_params

    def _calc_model_delta(self, model_pre, model_post):
        """计算模型参数变化量"""
        total_delta = 0.0
        for key in model_pre:
            delta = torch.norm(model_post[key] - model_pre[key]).item()
            total_delta += delta
        return total_delta

    def _finalize_training(self):
        """训练结束处理：恢复最佳模型并计算平均指标"""
        # 1. 恢复最佳模型
        if self.best_models:
            logging.info(
                f"恢复第 {self.best_models['epoch']+1} 轮的最佳模型 "
                f"(F1={self.best_models['global_f1']:.4f}, "
                f"Precision={self.best_models['global_precision']:.4f}, "
                f"AUC={self.best_models['global_auc']:.4f}, "  # 新增AUC显示
                f"Q1 Loss={self.best_models['global_q1_loss']:.4f})"
            )
            for i, algo in enumerate(self.algorithms):
                algo.trainer.policy.load_state_dict(self.best_models['client_models'][i]['policy'])
                algo.trainer.qf1.load_state_dict(self.best_models['client_models'][i]['qf1'])
                algo.trainer.qf2.load_state_dict(self.best_models['client_models'][i]['qf2'])
        
        # 2. 计算平均指标
        avg_metrics = self._calculate_average_metrics()
        
        # 3. 输出最终报告
        self._generate_final_report(avg_metrics)
        
    def _calculate_average_metrics(self):
        """计算所有轮次的平均指标"""
        avg_metrics = {
            'global': {},
            'clients': {}
        }
        
        # 计算全局平均
        for metric in self.metrics['global']:
            values = self.metrics['global'][metric]
            if values:
                avg_metrics['global'][metric] = np.mean(values)
        
        # 计算各客户端平均
        for client_id in self.metrics['clients']:
            client_avg = {}
            for metric in self.metrics['clients'][client_id]:
                values = self.metrics['clients'][client_id][metric]
                if values:
                    client_avg[metric] = np.mean(values)
            avg_metrics['clients'][client_id] = client_avg
        
        return avg_metrics

    def _generate_final_report(self, avg_metrics):
        """生成最终训练报告"""
        logging.info("\n" + "="*60)
        logging.info("联邦训练最终报告")
        logging.info(f"总训练轮次: {self.final_epoch}")
        logging.info(f"早停触发: {self.epochs_without_improvement >= self.patience}")
        
        # ===== 修复：显示完整全局指标（包含AUC）=====
        logging.info("\n全局平均指标:")
        metric_names = {
            'accuracy': 'Accuracy',
            'q1_loss': 'Q1 Loss',
            'q2_loss': 'Q2 Loss', 
            'policy_loss': 'Policy Loss',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1-Score',
            'auc': 'AUC'  # 新增AUC显示
        }
        
        for metric, value in avg_metrics['global'].items():
            display_name = metric_names.get(metric, metric.capitalize())
            logging.info(f"{display_name}: {value:.4f}")
        
        # ===== 修复：显示完整客户端指标（包含AUC）=====
        for client_id, metrics in avg_metrics['clients'].items():
            logging.info(f"\n客户端 {client_id} 平均指标:")
            for metric, value in metrics.items():
                display_name = metric_names.get(metric, metric.capitalize())
                logging.info(f"{display_name}: {value:.4f}")
        
        # ========== 新增：AUC专项统计报告 ==========
        logging.info("\n" + "="*40)
        logging.info("AUC专项统计报告")
        logging.info("="*40)
        
        # 全局AUC统计
        global_auc_values = self.metrics['global']['auc']
        if global_auc_values:
            avg_auc = np.mean(global_auc_values)
            max_auc = np.max(global_auc_values)
            min_auc = np.min(global_auc_values)
            std_auc = np.std(global_auc_values)
            
            logging.info(f"全局AUC统计:")
            logging.info(f"  平均AUC: {avg_auc:.4f}")
            logging.info(f"  最高AUC: {max_auc:.4f}")
            logging.info(f"  最低AUC: {min_auc:.4f}")
            logging.info(f"  标准差: {std_auc:.4f}")
        
        # 各客户端AUC统计
        logging.info(f"\n各客户端AUC对比:")
        client_auc_summary = []
        for client_id in self.metrics['clients']:
            client_auc_values = self.metrics['clients'][client_id]['auc']
            if client_auc_values:
                avg_client_auc = np.mean(client_auc_values)
                max_client_auc = np.max(client_auc_values)
                client_auc_summary.append((client_id, avg_client_auc, max_client_auc))
                logging.info(f"  客户端{client_id}: 平均AUC={avg_client_auc:.4f}, 最高AUC={max_client_auc:.4f}")
        
        # 排序并显示AUC性能排名
        if client_auc_summary:
            client_auc_summary.sort(key=lambda x: x[1], reverse=True)  # 按平均AUC排序
            logging.info(f"\nAUC性能排名（按平均AUC）:")
            for rank, (client_id, avg_auc, max_auc) in enumerate(client_auc_summary, 1):
                logging.info(f"  第{rank}名: 客户端{client_id} (平均AUC: {avg_auc:.4f})")
        
        logging.info("="*60)

    def _fuse_transformers(self):
        """FedFormer专用的Transformer聚合方法"""
        # 如果使用FedFormer，这里应该实现Transformer编码器的聚合逻辑
        # 暂时使用FedAvg作为占位符
        logging.info("执行FedFormer聚合（当前使用FedAvg实现）")
        self._fedavg()

    def get_global_metrics(self):
        """获取全局指标用于外部分析"""
        return self.metrics['global']
    
    def get_client_metrics(self, client_id):
        """获取指定客户端的指标"""
        return self.metrics['clients'].get(client_id, {})
    
    # ========== 新增：AUC专项分析方法 ==========
    def get_auc_analysis(self):
        """获取详细的AUC分析报告"""
        analysis = {
            'global_auc': {
                'all_values': self.metrics['global']['auc'],
                'mean': np.mean(self.metrics['global']['auc']) if self.metrics['global']['auc'] else 0.0,
                'std': np.std(self.metrics['global']['auc']) if self.metrics['global']['auc'] else 0.0,
                'max': np.max(self.metrics['global']['auc']) if self.metrics['global']['auc'] else 0.0,
                'min': np.min(self.metrics['global']['auc']) if self.metrics['global']['auc'] else 0.0
            },
            'client_auc': {}
        }
        
        for client_id in self.metrics['clients']:
            client_auc_values = self.metrics['clients'][client_id]['auc']
            analysis['client_auc'][client_id] = {
                'all_values': client_auc_values,
                'mean': np.mean(client_auc_values) if client_auc_values else 0.0,
                'std': np.std(client_auc_values) if client_auc_values else 0.0,
                'max': np.max(client_auc_values) if client_auc_values else 0.0,
                'min': np.min(client_auc_values) if client_auc_values else 0.0
            }
        
        return analysis