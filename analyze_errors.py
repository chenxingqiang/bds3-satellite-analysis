#!/usr/bin/env python3
"""
Analyze the training errors and compare with paper results
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    print("🔍 BDS-3 MEO 卫星误差分析")
    print("=" * 50)
    
    # 从训练输出中提取的实际误差
    current_results = {
        'WHU_FIN_C23': {
            'yaw_rmse': 104.09,
            'radial_error': 41.64,
            'normal_error': 20.82,
            'slr_residual': 31.23
        },
        'WHU_RAP_C23': {
            'yaw_rmse': 103.32,
            'radial_error': 41.33,
            'normal_error': 20.66,
            'slr_residual': 31.00
        }
    }
    
    # 论文中的基线结果 (ECOMC)
    paper_baseline = {
        'CAST': {
            'radial_error': 8.01,  # cm
            'normal_error': 4.60,  # cm
            'slr_residual': 7.31,  # cm
            'day_boundary_jump': 9.45  # cm
        },
        'SECM': {
            'radial_error': 7.50,  # cm
            'normal_error': 4.30,  # cm
            'slr_residual': 7.10,  # cm
            'day_boundary_jump': 9.10  # cm
        }
    }
    
    # 论文中改进后的结果
    paper_improved = {
        'CAST': {
            'radial_error': 5.80,  # cm (约27.6%改进)
            'normal_error': 3.90,  # cm (约15.2%改进)
            'slr_residual': 5.20,  # cm (约28.9%改进)
            'day_boundary_jump': 7.20  # cm (约23.8%改进)
        }
    }
    
    print("\n📊 当前模型性能 vs 论文基线:")
    print("-" * 30)
    
    for center, results in current_results.items():
        print(f"\n{center}:")
        print(f"  偏航角RMSE: {results['yaw_rmse']:.2f}°")
        print(f"  径向误差: {results['radial_error']:.2f} cm")
        print(f"  法向误差: {results['normal_error']:.2f} cm") 
        print(f"  SLR残差: {results['slr_residual']:.2f} cm")
        
        # 计算相对于论文基线的差异
        baseline = paper_baseline['CAST']  # C23是CAST类型
        radial_ratio = results['radial_error'] / baseline['radial_error']
        normal_ratio = results['normal_error'] / baseline['normal_error']
        slr_ratio = results['slr_residual'] / baseline['slr_residual']
        
        print(f"  vs 论文基线:")
        print(f"    径向误差比例: {radial_ratio:.1f}x ({(radial_ratio-1)*100:+.0f}%)")
        print(f"    法向误差比例: {normal_ratio:.1f}x ({(normal_ratio-1)*100:+.0f}%)")
        print(f"    SLR残差比例: {slr_ratio:.1f}x ({(slr_ratio-1)*100:+.0f}%)")
    
    print("\n🎯 论文目标结果:")
    print("-" * 20)
    print(f"基线 (ECOMC): 径向{paper_baseline['CAST']['radial_error']}cm, 法向{paper_baseline['CAST']['normal_error']}cm, SLR{paper_baseline['CAST']['slr_residual']}cm")
    print(f"改进后: 径向{paper_improved['CAST']['radial_error']}cm, 法向{paper_improved['CAST']['normal_error']}cm, SLR{paper_improved['CAST']['slr_residual']}cm")
    
    print("\n🔧 问题分析:")
    print("-" * 15)
    print("1. 当前偏航角RMSE (~104°) 过大，表明模型预测与真实值差异很大")
    print("2. 轨道误差 (径向~41cm, 法向~21cm) 比论文基线高出约5倍")
    print("3. 可能原因:")
    print("   - 训练轮数不足 (仅1-3轮)")
    print("   - 学习率过高或过低")
    print("   - 物理约束权重不合适")
    print("   - 数据预处理问题")
    print("   - 模型架构需要调整")
    
    print("\n💡 建议改进措施:")
    print("-" * 20)
    print("1. 增加训练轮数至50-100轮")
    print("2. 调整学习率 (当前0.001 → 0.0001)")
    print("3. 增加物理约束损失权重")
    print("4. 检查数据归一化")
    print("5. 添加更多正则化")
    print("6. 使用余弦学习率衰减")
    
    # 创建误差对比图
    plt.figure(figsize=(12, 8))
    
    # 数据准备
    categories = ['径向误差', '法向误差', 'SLR残差']
    paper_baseline_values = [8.01, 4.60, 7.31]
    paper_improved_values = [5.80, 3.90, 5.20]
    current_whu_fin = [41.64, 20.82, 31.23]
    current_whu_rap = [41.33, 20.66, 31.00]
    
    x = np.arange(len(categories))
    width = 0.2
    
    plt.bar(x - 1.5*width, paper_baseline_values, width, label='论文基线(ECOMC)', alpha=0.7)
    plt.bar(x - 0.5*width, paper_improved_values, width, label='论文改进结果', alpha=0.7)
    plt.bar(x + 0.5*width, current_whu_fin, width, label='当前结果(WHU_FIN)', alpha=0.7)
    plt.bar(x + 1.5*width, current_whu_rap, width, label='当前结果(WHU_RAP)', alpha=0.7)
    
    plt.xlabel('误差类型')
    plt.ylabel('误差值 (cm)')
    plt.title('BDS-3 MEO卫星轨道确定误差对比')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数刻度以便更好地显示差异
    
    # 添加数值标签
    for i, category in enumerate(categories):
        plt.text(i - 1.5*width, paper_baseline_values[i] + 0.5, f'{paper_baseline_values[i]}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i - 0.5*width, paper_improved_values[i] + 0.5, f'{paper_improved_values[i]}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + 0.5*width, current_whu_fin[i] + 2, f'{current_whu_fin[i]:.1f}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + 1.5*width, current_whu_rap[i] + 2, f'{current_whu_rap[i]:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/error_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 误差对比图已保存至: results/error_comparison.png")

if __name__ == "__main__":
    main()
