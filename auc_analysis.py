# auc_analysis.py - AUC指标分析脚本
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fed_algorithm import FedAlgorithm

def analyze_auc_results(fed_algorithm):
    """
    详细分析联邦学习中的AUC指标
    
    Args:
        fed_algorithm: 训练完成的FedAlgorithm实例
    """
    
    # 获取AUC分析数据
    auc_analysis = fed_algorithm.get_auc_analysis()
    
    print("="*60)
    print("联邦学习AUC指标详细分析报告")
    print("="*60)
    
    # 1. 全局AUC分析
    global_auc = auc_analysis['global_auc']
    print(f"\n1. 全局AUC统计:")
    print(f"   平均AUC: {global_auc['mean']:.4f}")
    print(f"   标准差: {global_auc['std']:.4f}")
    print(f"   最高AUC: {global_auc['max']:.4f}")
    print(f"   最低AUC: {global_auc['min']:.4f}")
    print(f"   总轮次: {len(global_auc['all_values'])}")
    
    # 2. 各客户端AUC对比
    print(f"\n2. 各客户端AUC对比:")
    client_auc_summary = []
    for client_id, client_data in auc_analysis['client_auc'].items():
        client_auc_summary.append({
            'client_id': client_id,
            'mean_auc': client_data['mean'],
            'std_auc': client_data['std'],
            'max_auc': client_data['max'],
            'min_auc': client_data['min']
        })
        print(f"   客户端{client_id}: 平均={client_data['mean']:.4f}, "
              f"标准差={client_data['std']:.4f}, 最高={client_data['max']:.4f}")
    
    # 3. AUC性能排名
    client_auc_summary.sort(key=lambda x: x['mean_auc'], reverse=True)
    print(f"\n3. AUC性能排名:")
    for rank, client_info in enumerate(client_auc_summary, 1):
        print(f"   第{rank}名: 客户端{client_info['client_id']} "
              f"(平均AUC: {client_info['mean_auc']:.4f})")
    
    # 4. AUC收敛性分析
    print(f"\n4. AUC收敛性分析:")
    if len(global_auc['all_values']) >= 10:
        # 计算前半段和后半段的AUC均值
        mid_point = len(global_auc['all_values']) // 2
        early_auc = np.mean(global_auc['all_values'][:mid_point])
        late_auc = np.mean(global_auc['all_values'][mid_point:])
        improvement = late_auc - early_auc
        
        print(f"   前半段平均AUC: {early_auc:.4f}")
        print(f"   后半段平均AUC: {late_auc:.4f}")
        print(f"   整体改善: {improvement:.4f}")
        
        if improvement > 0.01:
            print("   ✓ AUC呈现良好的上升趋势")
        elif improvement > -0.01:
            print("   ○ AUC基本稳定")
        else:
            print("   ✗ AUC出现下降趋势，需要检查")
    
    return auc_analysis

def plot_auc_trends(fed_algorithm, save_path=None):
    """
    绘制AUC趋势图
    
    Args:
        fed_algorithm: 训练完成的FedAlgorithm实例
        save_path: 图片保存路径（可选）
    """
    
    # 获取指标数据
    global_metrics = fed_algorithm.get_global_metrics()
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('联邦学习AUC指标分析', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(global_metrics['auc']) + 1)
    
    # 1. 全局AUC趋势
    axes[0, 0].plot(epochs, global_metrics['auc'], 'b-', linewidth=2, marker='o')
    axes[0, 0].set_title('全局AUC趋势')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # 2. 各客户端AUC对比
    for client_id in range(len(fed_algorithm.algorithms)):
        client_metrics = fed_algorithm.get_client_metrics(client_id)
        if 'auc' in client_metrics and client_metrics['auc']:
            axes[0, 1].plot(epochs, client_metrics['auc'], 
                          marker='o', label=f'客户端{client_id}', alpha=0.7)
    
    axes[0, 1].set_title('各客户端AUC趋势对比')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # 3. AUC分布箱线图
    client_auc_data = []
    client_labels = []
    for client_id in range(len(fed_algorithm.algorithms)):
        client_metrics = fed_algorithm.get_client_metrics(client_id)
        if 'auc' in client_metrics and client_metrics['auc']:
            client_auc_data.append(client_metrics['auc'])
            client_labels.append(f'客户端{client_id}')
    
    if client_auc_data:
        axes[1, 0].boxplot(client_auc_data, labels=client_labels)
        axes[1, 0].set_title('各客户端AUC分布')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # 4. F1 vs AUC散点图
    axes[1, 1].scatter(global_metrics['f1'], global_metrics['auc'], 
                      c=epochs, cmap='viridis', s=50, alpha=0.7)
    axes[1, 1].set_title('F1 vs AUC关系')
    axes[1, 1].set_xlabel('F1 Score')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加颜色条
    colorbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    colorbar.set_label('训练轮次')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"AUC分析图已保存至: {save_path}")
    
    plt.show()

def generate_auc_report(fed_algorithm, report_path="auc_analysis_report.txt"):
    """
    生成详细的AUC分析报告文件
    
    Args:
        fed_algorithm: 训练完成的FedAlgorithm实例
        report_path: 报告文件保存路径
    """
    
    auc_analysis = fed_algorithm.get_auc_analysis()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("联邦学习AUC指标详细分析报告\n")
        f.write("="*60 + "\n\n")
        
        # 训练概况
        f.write("训练概况:\n")
        f.write(f"  训练轮次: {len(auc_analysis['global_auc']['all_values'])}\n")
        f.write(f"  客户端数量: {len(auc_analysis['client_auc'])}\n")
        f.write(f"  全局平均AUC: {auc_analysis['global_auc']['mean']:.4f}\n\n")
        
        # 全局AUC统计
        f.write("全局AUC统计:\n")
        global_auc = auc_analysis['global_auc']
        f.write(f"  平均值: {global_auc['mean']:.4f}\n")
        f.write(f"  标准差: {global_auc['std']:.4f}\n")
        f.write(f"  最大值: {global_auc['max']:.4f}\n")
        f.write(f"  最小值: {global_auc['min']:.4f}\n")
        f.write(f"  变异系数: {global_auc['std']/global_auc['mean']:.4f}\n\n")
        
        # 各客户端详细统计
        f.write("各客户端AUC详细统计:\n")
        for client_id, client_data in auc_analysis['client_auc'].items():
            f.write(f"\n  客户端{client_id}:\n")
            f.write(f"    平均AUC: {client_data['mean']:.4f}\n")
            f.write(f"    标准差: {client_data['std']:.4f}\n")
            f.write(f"    最高AUC: {client_data['max']:.4f}\n")
            f.write(f"    最低AUC: {client_data['min']:.4f}\n")
            f.write(f"    稳定性: {1 - client_data['std']/client_data['mean']:.4f}\n")
        
        # 性能对比
        f.write("\n客户端性能排名:\n")
        client_rankings = sorted(auc_analysis['client_auc'].items(), 
                               key=lambda x: x[1]['mean'], reverse=True)
        for rank, (client_id, client_data) in enumerate(client_rankings, 1):
            f.write(f"  第{rank}名: 客户端{client_id} "
                   f"(平均AUC: {client_data['mean']:.4f})\n")
        
        f.write(f"\n报告生成完成: {report_path}\n")
    
    print(f"详细AUC分析报告已保存至: {report_path}")

# 使用示例
if __name__ == "__main__":
    # 注意：这里假设你已经有一个训练完成的 fed_algorithm 实例
    # 在实际使用中，你需要在 main.py 的 experiment 函数结束时调用这些分析函数
    
    print("AUC分析工具已准备就绪!")
    print("\n使用方法:")
    print("1. 在训练完成后调用: analyze_auc_results(fed_algorithm)")
    print("2. 绘制AUC趋势图: plot_auc_trends(fed_algorithm, 'auc_plots.png')")
    print("3. 生成详细报告: generate_auc_report(fed_algorithm)")
    print("\n请在main.py中的experiment函数末尾添加这些调用。")