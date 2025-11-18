
import os
import logging
import torch
import click
import pandas as pd
import numpy as np
import rlkit.torch.pytorch_util as ptu
from data_loader import IntrusionDataLoader
from policy import ClassificationPolicy
from sac_trainer import SACClassifierTrainer
from fed_algorithm import FedAlgorithm
from intrusion_detection_env import IntrusionDetectionEnv 
from networks import ConcatMlp
from sac_algorithm import TorchBatchRLAlgorithm
from simple_replay_buffer import SimpleReplayBuffer
from fed_path_collector import FedPathCollector

from auc_analysis import analyze_auc_results, plot_auc_trends, generate_auc_report


TARGET_COLUMN = "Label"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_q_networks(obs_dim, action_dim=2, hidden_sizes=[256,256], device='cpu'):
    input_dim = obs_dim + action_dim
    qf1 = ConcatMlp(input_dim, 1, hidden_sizes).to(device)
    qf2 = ConcatMlp(input_dim, 1, hidden_sizes).to(device)
    target_qf1 = ConcatMlp(input_dim, 1, hidden_sizes).to(device)
    target_qf2 = ConcatMlp(input_dim, 1, hidden_sizes).to(device)
    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())
    return qf1, qf2, target_qf1, target_qf2

def experiment(variant):

    if torch.cuda.is_available():
        ptu.set_gpu_mode(True)
        logging.info(f"使用GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        ptu.set_gpu_mode(False)
        logging.info("使用CPU设备")
    
    device = ptu.device
    algorithms = []
    

    try:
        logging.info("开始特征维度验证...")
        dummy_loader = IntrusionDataLoader(
            client_id=0,
            data_root=variant['data_root'],
            batch_size=variant['batch_size']
        )
        obs_dim = dummy_loader.feature_dim
        logging.info(f"成功获取特征维度: {obs_dim}")
        

        features, labels = dummy_loader.sample_batch()
        logging.info(f"采样特征形状: {features.shape}, 数据类型: {features.dtype}")
        logging.info(f"采样标签形状: {labels.shape}, 数据类型: {labels.dtype}")
        
 
        if features.shape[1] != obs_dim:
            raise ValueError(f"特征维度不匹配! 预期: {obs_dim}, 实际: {features.shape[1]}")
            
    except Exception as e:
        logging.error(f"无法获取特征维度: {str(e)}")
        raise
  
    try:
        logging.info("开始验证客户端数据标签分布...")
        for client_id in range(variant['num_agents']):
            train_path = os.path.join(variant['data_root'], f"client_{client_id}/train.csv")
            if not os.path.exists(train_path):
                logging.warning(f"客户端{client_id} 训练文件不存在")
                continue
                
            client_train = pd.read_csv(train_path)
    
            if TARGET_COLUMN not in client_train.columns:
                raise ValueError(f"客户端{client_id} 数据缺少标签列 {TARGET_COLUMN}")
            
    
            label_values = client_train[TARGET_COLUMN].unique()
            invalid_values = [v for v in label_values if v not in (0, 1)]
            if invalid_values:
                raise ValueError(f"客户端{client_id} 包含非法标签值: {invalid_values}")
            
            label_counts = client_train[TARGET_COLUMN].value_counts()
            imbalance_ratio = label_counts.max() / label_counts.min()
            
            logging.info(f"客户端{client_id} 标签分布:\n{label_counts.to_string()}")
            if imbalance_ratio > 10:
                logging.warning(f"客户端{client_id} 标签严重不平衡 (比例: {imbalance_ratio:.1f}:1)")
    except Exception as e:
        logging.error(f"数据验证失败: {str(e)}")
        raise
    
    for client_id in range(variant['num_agents']):
        try:
            data_loader = IntrusionDataLoader(
                client_id=client_id,
                data_root=variant['data_root'],
                batch_size=variant['batch_size']
            )
            
     
            features, labels = data_loader.sample_batch()
            
      
            if not isinstance(features, np.ndarray):
                raise TypeError(f"特征应为NumPy数组, 实际类型: {type(features)}")
            if features.dtype != np.float32:
                raise TypeError(f"特征应为float32, 实际类型: {features.dtype}")
            
    
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"标签应为NumPy数组, 实际类型: {type(labels)}")
            if labels.dtype != np.int64:
                raise TypeError(f"标签应为int64, 实际类型: {labels.dtype}")
                
            logging.info(f"客户端{client_id} 数据验证通过")
        except Exception as e:
            logging.error(f"客户端{client_id} 数据验证失败: {str(e)}")
          
            try:
                problem_path = os.path.join(variant['data_root'], f"client_{client_id}/train.csv")
                logging.error(f"问题数据路径: {problem_path}")
                if os.path.exists(problem_path):
                    problem_df = pd.read_csv(problem_path)
                    logging.error(f"数据列类型:\n{problem_df.dtypes}")
                    logging.error(f"前5行数据:\n{problem_df.head().to_string()}")
            except Exception as e:
                logging.error(f"无法记录问题数据详情: {str(e)}")
            raise
    

    logging.info("="*50)
    logging.info("开始端到端维度验证")
    
    try:
    
        test_loader = IntrusionDataLoader(
            client_id=0,
            data_root=variant['data_root'],
            batch_size=variant['batch_size']
        )
        

        test_env = IntrusionDetectionEnv(
            data_loader=test_loader,
            max_steps=variant['max_steps']
        )
        

        obs = test_env.reset()
        logging.info(f"环境重置返回观察值形状: {obs.shape}")
        

        if obs.shape[0] != obs_dim:
            raise ValueError(f"环境返回观察值维度错误: 预期 {obs_dim}, 实际 {obs.shape[0]}")
        

        action = 0
        next_obs, reward, done, _ = test_env.step(action)
        logging.info(f"环境步骤返回下一观察值形状: {next_obs.shape}")
        

        if next_obs.shape[0] != obs_dim:
            raise ValueError(f"环境返回下一观察值维度错误: 预期 {obs_dim}, 实际 {next_obs.shape[0]}")
        

        test_policy = ClassificationPolicy(
            obs_dim=obs_dim,
            hidden_sizes=variant['policy_hidden_sizes'],
            layer_normalization=variant['layer_norm']
        ).to(device)
        

        action, log_prob = test_policy.get_action(obs)
        logging.info(f"策略网络返回动作: {action}, 日志概率: {log_prob}")
        
   
        logging.info("测试环境spec属性...")
        env_spec = test_env.spec
        logging.info(f"环境spec - 观察空间: {env_spec.observation_space.shape}")
        logging.info(f"环境spec - 动作空间: {env_spec.action_space.n}")
        logging.info(f"环境spec - 最大步数: {env_spec.max_episode_steps}")
        
     
        logging.info("测试路径收集器接口...")
        test_collector = FedPathCollector(
            policy=test_policy,
            device=device
          
        )
        test_collector.set_data_loader(test_loader)
        

        test_paths = test_collector.collect_new_paths(
            max_path_length=10,
            num_steps=20
        )
        logging.info(f"成功收集测试路径: {len(test_paths)} 条")
        
        logging.info("端到端维度验证成功")
    except Exception as e:
        logging.error(f"端到端维度验证失败: {str(e)}")
        raise
    finally:
        logging.info("="*50)
    

    for client_id in range(variant['num_agents']):
        try:
            logging.info(f"初始化客户端 {client_id} 算法...")
            
            data_loader = IntrusionDataLoader(
                client_id=client_id,
                data_root=variant['data_root'],
                batch_size=variant['batch_size']
            )
            env = IntrusionDetectionEnv(
                data_loader=data_loader,
                max_steps=variant['max_steps']
            )
            
            policy = ClassificationPolicy(
                obs_dim=obs_dim,
                hidden_sizes=variant['policy_hidden_sizes'],
                layer_normalization=variant['layer_norm']
            ).to(device)
            
            qf1, qf2, target_qf1, target_qf2 = initialize_q_networks(obs_dim, device=device)
            

            label_counts = data_loader.train_df[TARGET_COLUMN].value_counts()
            minority_class = 1 if label_counts[0] > label_counts[1] else 0
            majority_weight = 1.0
            minority_weight = max(5.0, label_counts.max() / label_counts.min())
            
            logging.info(f"客户端{client_id} 类别权重: 0={majority_weight:.1f}, 1={minority_weight:.1f}")
            
            trainer = SACClassifierTrainer(
                client_id,
                env,
                policy,
                qf1,
                qf2,
                target_qf1,
                target_qf2,
                cls_weights=torch.tensor([majority_weight, minority_weight], device=device)
            )
            
     
            expl_collector = FedPathCollector(
                policy=policy,
                device=device
              
            )
            expl_collector.set_data_loader(data_loader)
            
            eval_collector = FedPathCollector(
                policy=policy,
                device=device
             
            )
            eval_collector.set_data_loader(data_loader)
            
            algorithm = TorchBatchRLAlgorithm(
                client_id=client_id,
                trainer=trainer,
                exploration_env=env,
                evaluation_env=env,
                exploration_data_collector=expl_collector,
                evaluation_data_collector=eval_collector,
                replay_buffer=SimpleReplayBuffer(
                    max_size=variant["replay_buffer_size"],
                    max_path_length=variant['algorithm_kwargs']['max_path_length'],
                    env_spec=env.spec  
                ),
                batch_size=variant["batch_size"],
                num_epochs=variant["num_epochs"],
                **variant['algorithm_kwargs']
            )
            algorithms.append(algorithm)
            logging.info(f"客户端 {client_id} 初始化成功")
        except Exception as e:
            logging.error(f"客户端 {client_id} 初始化失败: {str(e)}")
            raise
    

    logging.info("开始联邦训练...")
    fed_algorithm = FedAlgorithm(algorithms, variant['num_epochs'], fedFormer=False, patience=5)
    fed_algorithm.train()
    logging.info("联邦训练完成")
    

    logging.info("\n" + "="*60)
    logging.info("开始AUC详细分析...")
    logging.info("="*60)
    
    try:

        auc_analysis = analyze_auc_results(fed_algorithm)
        
    
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            plot_save_path = "results/auc_analysis_plots.png"
            os.makedirs("results", exist_ok=True)
            plot_auc_trends(fed_algorithm, save_path=plot_save_path)
            logging.info(f"AUC趋势图已保存至: {plot_save_path}")
        except ImportError:
            logging.warning("matplotlib未安装，跳过图形生成")
        except Exception as e:
            logging.warning(f"生成AUC图表时出错: {str(e)}")
        
 
        report_save_path = "results/auc_detailed_report.txt"
        os.makedirs("results", exist_ok=True)
        generate_auc_report(fed_algorithm, report_path=report_save_path)
        
  
        global_auc_mean = auc_analysis['global_auc']['mean']
        global_auc_std = auc_analysis['global_auc']['std']
        
        logging.info(f"\n 联邦学习AUC性能摘要:")
        logging.info(f"   全局平均AUC: {global_auc_mean:.4f} ± {global_auc_std:.4f}")
        

        best_client = max(auc_analysis['client_auc'].items(), 
                         key=lambda x: x[1]['mean'])
        worst_client = min(auc_analysis['client_auc'].items(), 
                          key=lambda x: x[1]['mean'])
        
        logging.info(f"   最佳客户端: 客户端{best_client[0]} (AUC: {best_client[1]['mean']:.4f})")
        logging.info(f"   最差客户端: 客户端{worst_client[0]} (AUC: {worst_client[1]['mean']:.4f})")
        logging.info(f"   性能差距: {best_client[1]['mean'] - worst_client[1]['mean']:.4f}")
        
        # AUC质量评估
        if global_auc_mean >= 0.9:
            auc_quality = "优秀 "
        elif global_auc_mean >= 0.8:
            auc_quality = "良好 ✓"
        elif global_auc_mean >= 0.7:
            auc_quality = "中等 ○"
        else:
            auc_quality = "需要改进 "
        
        logging.info(f"   AUC质量评估: {auc_quality}")
        
    except Exception as e:
        logging.error(f"AUC分析过程中出错: {str(e)}")
        logging.error("继续程序执行，但AUC分析可能不完整")
    
    logging.info("="*60)
    logging.info("训练和分析全部完成!")
    logging.info("="*60)

@click.command()
@click.option("--data_root", default="data/clients")  # 指向预处理后的客户端目录
@click.option("--num_agents", default=4)
def main(data_root, num_agents):

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    variant = dict(
        data_root=data_root,
        num_agents=num_agents,
        batch_size=128,
        policy_hidden_sizes=[256,256],
        layer_norm=True,
        replay_buffer_size=int(1e5),
        num_epochs=50,
        max_steps=1000,
        algorithm_kwargs=dict(
            max_path_length=1000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=1000,
        )
    )
    
    logging.info("开始入侵检测联邦训练")
    logging.info(f"配置参数: {variant}")
    
    try:
        experiment(variant)
        logging.info("训练成功完成")
    except Exception as e:
        logging.error(f"训练失败: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
