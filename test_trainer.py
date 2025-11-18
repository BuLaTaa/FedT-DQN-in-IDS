# test_trainer.py - 测试训练器修复
import torch
import numpy as np
import logging
import sys
import os

# 添加项目路径
sys.path.append('.')

from sac_trainer import SACClassifierTrainer
from policy import ClassificationPolicy
from networks import ConcatMlp
import rlkit.torch.pytorch_util as ptu

# 设置日志
logging.basicConfig(level=logging.DEBUG)

def test_trainer_fix():
    """测试训练器是否能正确处理张量输入"""
    
    print("🧪 开始测试 SACClassifierTrainer 修复...")
    
    # 设置设备
    if torch.cuda.is_available():
        ptu.set_gpu_mode(True)
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        ptu.set_gpu_mode(False) 
        device = torch.device('cpu')
        print("✅ 使用CPU")
    
    # 创建测试数据
    batch_size = 4
    obs_dim = 10
    
    # 创建张量格式的批次数据（模拟 ReplayBuffer 的输出）
    tensor_batch = {
        'observations': torch.randn(batch_size, obs_dim, dtype=torch.float32, device=device),
        'actions': torch.randint(0, 2, (batch_size,), dtype=torch.long, device=device),
        'rewards': torch.randn(batch_size, dtype=torch.float32, device=device),
        'next_observations': torch.randn(batch_size, obs_dim, dtype=torch.float32, device=device),
        'terminals': torch.randint(0, 2, (batch_size,), dtype=torch.bool, device=device),
        'labels': torch.randint(0, 2, (batch_size,), dtype=torch.long, device=device),
    }
    
    print("✅ 创建测试数据:")
    for key, value in tensor_batch.items():
        print(f"  {key}: {value.shape}, {value.dtype}, {value.device}")
    
    # 创建策略网络和Q网络
    try:
        policy = ClassificationPolicy(
            obs_dim=obs_dim,
            hidden_sizes=[32, 32],
            layer_normalization=False
        ).to(device)
        
        qf1 = ConcatMlp(obs_dim + 2, 1, [32, 32]).to(device)
        qf2 = ConcatMlp(obs_dim + 2, 1, [32, 32]).to(device)
        target_qf1 = ConcatMlp(obs_dim + 2, 1, [32, 32]).to(device)
        target_qf2 = ConcatMlp(obs_dim + 2, 1, [32, 32]).to(device)
        
        target_qf1.load_state_dict(qf1.state_dict())
        target_qf2.load_state_dict(qf2.state_dict())
        
        print("✅ 创建网络成功")
        
    except Exception as e:
        print(f"❌ 创建网络失败: {str(e)}")
        return False
    
    # 创建训练器
    try:
        trainer = SACClassifierTrainer(
            client_id=0,
            env=None,  # 测试时不需要环境
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            cls_weights=torch.tensor([1.0, 1.0], device=device)
        )
        
        print("✅ 创建训练器成功")
        
    except Exception as e:
        print(f"❌ 创建训练器失败: {str(e)}")
        return False
    
    # 测试张量输入
    try:
        print("\n🔥 测试张量输入训练...")
        losses = trainer.train(tensor_batch)
        
        print("✅ 张量输入训练成功!")
        print("📊 损失信息:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ 张量输入训练失败: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False
    
    # 测试NumPy输入（为了兼容性）
    try:
        print("\n🔥 测试NumPy输入训练...")
        numpy_batch = {}
        for key, value in tensor_batch.items():
            if isinstance(value, torch.Tensor):
                numpy_batch[key] = value.detach().cpu().numpy()
            else:
                numpy_batch[key] = value
                
        losses = trainer.train(numpy_batch)
        
        print("✅ NumPy输入训练也成功!")
        print("📊 损失信息:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ NumPy输入训练失败: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False
    
    print("\n🎉 所有测试通过! SACClassifierTrainer 修复成功!")
    return True

if __name__ == "__main__":
    success = test_trainer_fix()
    if success:
        print("\n✅ 测试完成，可以运行 main.py 了!")
    else:
        print("\n❌ 测试失败，需要进一步修复!")