#!/usr/bin/env python3
"""
BDS-3 MEO Satellite Yaw Attitude Analysis - Figure 3 Reproduction
Comparison of quaternion yaw angle, nominal yaw angle and model yaw angle for satellite midnight maneuver

This script reproduces Figure 3 from the paper:
"BDS-3 MEO satellite yaw attitude and improving solar radiation pressure model during eclipse seasons"

Author: Claude (based on Li Hui et al. research)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import re
from typing import Dict, List, Tuple
import matplotlib.patches as patches

# 设置字体（避免中文字体问题）
plt.rcParams['axes.unicode_minus'] = False

class OBXParser:
    """OBX格式文件解析器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.satellites = {}
        self.epoch_interval = 30.0  # seconds
        
    def parse(self) -> Dict:
        """解析OBX文件，提取四元数数据"""
        data = {}
        
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # 找到数据开始位置
        data_start = False
        current_epoch = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('+EPHEMERIS/DATA'):
                data_start = True
                continue
                
            if not data_start:
                continue
                
            if line.startswith('-EPHEMERIS/DATA'):
                break
                
            # 解析时间行
            if line.startswith('##'):
                time_parts = line.split()
                year = int(time_parts[1])
                month = int(time_parts[2])
                day = int(time_parts[3])
                hour = int(time_parts[4])
                minute = int(time_parts[5])
                second = float(time_parts[6])
                
                current_epoch = datetime(year, month, day, hour, minute, int(second))
                continue
            
            # 解析ATT记录
            if line.startswith('ATT') and current_epoch:
                parts = line.split()
                if len(parts) >= 7:
                    sat_id = parts[1]
                    q0 = float(parts[3])  # scalar part
                    q1 = float(parts[4])  # x
                    q2 = float(parts[5])  # y
                    q3 = float(parts[6])  # z
                    
                    if sat_id not in data:
                        data[sat_id] = {'epochs': [], 'quaternions': []}
                    
                    data[sat_id]['epochs'].append(current_epoch)
                    data[sat_id]['quaternions'].append([q0, q1, q2, q3])
        
        return data

def quaternion_to_yaw(q: np.ndarray) -> float:
    """
    将四元数转换为偏航角
    q = [q0, q1, q2, q3] = [w, x, y, z]
    """
    q0, q1, q2, q3 = q
    
    # 四元数转欧拉角 (Roll, Pitch, Yaw)
    # Yaw (Z轴转角)
    yaw = np.arctan2(2.0 * (q0*q3 + q1*q2), 
                     1.0 - 2.0 * (q2*q2 + q3*q3))
    
    return np.degrees(yaw)

def calculate_orbital_angle(epoch: datetime, orbital_period: float = 12.87) -> float:
    """
    计算卫星轨道角 μ
    orbital_period: 轨道周期(小时)，BDS-3 MEO约为12.87小时
    """
    # 将时间转换为从午夜开始的小时数
    hours_from_midnight = epoch.hour + epoch.minute/60 + epoch.second/3600
    
    # 计算轨道角 (度)
    # 假设午夜对应轨道角0度或180度
    mu = (hours_from_midnight / orbital_period) * 360.0
    
    # 调整到[-180, 180]范围，子夜机动通常在轨道角180度附近
    if mu > 180:
        mu = mu - 360
    
    return mu

def calculate_nominal_yaw(mu_deg: float, beta_deg: float = 1.0) -> float:
    """
    计算名义偏航姿态角
    根据论文公式 (1): ψ_n = atan2(-tan(β), sin(μ))
    
    Args:
        mu_deg: 轨道角 (度)
        beta_deg: 太阳高度角 (度)
    """
    mu_rad = np.radians(mu_deg)
    beta_rad = np.radians(beta_deg)
    
    psi_n = np.arctan2(-np.tan(beta_rad), np.sin(mu_rad))
    
    return np.degrees(psi_n)

def calculate_whu_model_yaw(mu_deg: float, beta_deg: float = 1.0) -> float:
    """
    计算WHU模型偏航姿态角 (简化版本)
    在深度地影期会有特殊的机动策略
    """
    mu_rad = np.radians(mu_deg)
    beta_rad = np.radians(beta_deg)
    
    # WHU模型在机动期间的修正
    # 这里使用简化的模型，实际模型更复杂
    if abs(mu_deg) > 150:  # 接近子夜机动
        # 提前开始机动，更平滑的过渡
        transition_factor = (abs(mu_deg) - 150) / 30  # 30度范围内过渡
        transition_factor = min(transition_factor, 1.0)
        
        # 目标角度
        target_yaw = 180 if mu_deg > 0 else -180
        nominal_yaw = calculate_nominal_yaw(mu_deg, beta_deg)
        
        # 平滑过渡
        psi_whu = nominal_yaw + transition_factor * (target_yaw - nominal_yaw)
    else:
        psi_whu = calculate_nominal_yaw(mu_deg, beta_deg)
    
    return psi_whu

def calculate_csno_model_yaw(mu_deg: float, beta_deg: float = 1.0) -> float:
    """
    计算CSNO模型偏航姿态角 (简化版本)
    CSNO模型机动时间更早，过程更平滑
    """
    mu_rad = np.radians(mu_deg)
    beta_rad = np.radians(beta_deg)
    
    # CSNO模型提前更多开始机动
    if abs(mu_deg) > 120:  # 比WHU模型提前30度开始机动
        # 更长的过渡期
        transition_factor = (abs(mu_deg) - 120) / 60  # 60度范围内过渡
        transition_factor = min(transition_factor, 1.0)
        
        # 目标角度
        target_yaw = 180 if mu_deg > 0 else -180
        nominal_yaw = calculate_nominal_yaw(mu_deg, beta_deg)
        
        # 更平滑的过渡函数
        smooth_factor = 0.5 * (1 - np.cos(np.pi * transition_factor))
        psi_csno = nominal_yaw + smooth_factor * (target_yaw - nominal_yaw)
    else:
        psi_csno = calculate_nominal_yaw(mu_deg, beta_deg)
    
    return psi_csno

def filter_midnight_maneuver_data(epochs: List[datetime], quaternions: List, 
                                  target_hours: List[float] = [22, 23, 0, 1, 2]) -> Tuple[List, List]:
    """
    过滤出子夜机动期间的数据
    target_hours: 目标时间段(小时)
    """
    filtered_epochs = []
    filtered_quaternions = []
    
    for epoch, quat in zip(epochs, quaternions):
        hour = epoch.hour + epoch.minute/60 + epoch.second/3600
        
        # 检查是否在目标时间段内
        if any(abs(hour - th) < 1.5 or abs(hour - th + 24) < 1.5 or abs(hour - th - 24) < 1.5 
               for th in target_hours):
            filtered_epochs.append(epoch)
            filtered_quaternions.append(quat)
    
    return filtered_epochs, filtered_quaternions

def load_midnight_data(data_dir: str, satellites: List[str] = ['C23', 'C25']) -> Dict:
    """加载子夜机动期间的数据"""
    centers = ['WHU_FIN', 'WHU_RAP']
    all_data = {}
    
    for center in centers:
        center_dir = os.path.join(data_dir, center)
        all_data[center] = {}
        
        # 查找文件
        for file in os.listdir(center_dir):
            if file.endswith('.OBX'):
                file_path = os.path.join(center_dir, file)
                parser = OBXParser(file_path)
                data = parser.parse()
                
                # 提取目标卫星的子夜机动数据
                for sat in satellites:
                    if sat in data:
                        epochs, quaternions = filter_midnight_maneuver_data(
                            data[sat]['epochs'], data[sat]['quaternions'])
                        
                        if epochs:
                            if sat not in all_data[center]:
                                all_data[center][sat] = {'epochs': [], 'quaternions': []}
                            
                            all_data[center][sat]['epochs'].extend(epochs)
                            all_data[center][sat]['quaternions'].extend(quaternions)
    
    return all_data

def plot_figure3(all_data: Dict, save_path: str = None):
    """绘制图3：卫星子夜机动四元数偏航姿态角、名义偏航姿态角与常用模型偏航姿态角对比"""
    
    # 创建4x2子图布局 (4个分析中心 × 2个卫星)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    satellites = ['C23', 'C25']
    sat_names = ['C23', 'C25']
    centers = ['WHU-RAP', 'WHU-FIN', 'COD-FIN', 'GFZ-RAP']
    
    colors = {
        'QUA': '#FF6B35',      # 橙红色 - 四元数
        'NOM': '#0000FF',      # 蓝色 - 名义偏航
        'WHU': '#00FF00',      # 绿色 - WHU模型
        'CSNO': '#808080'      # 灰色 - CSNO模型
    }
    
    beta_angles = [1.2, 0.19]  # 论文中提到的太阳高度角范围
    
    for sat_idx, (sat, sat_name, beta) in enumerate(zip(satellites, sat_names, beta_angles)):
        for center_idx, center_name in enumerate(centers):
            ax = axes[sat_idx, center_idx]
            
            # 生成轨道角范围
            mu_theory = np.linspace(-20, 20, 200)
            
            # 名义偏航角
            yaw_nom = [calculate_nominal_yaw(mu, beta) for mu in mu_theory]
            ax.plot(mu_theory, yaw_nom, 'o', color=colors['NOM'], 
                   markersize=3, label='NOM', alpha=0.8)
            
            # WHU模型 (适用于C23)
            if sat == 'C23':
                yaw_whu = [calculate_whu_model_yaw(mu, beta) for mu in mu_theory]
                ax.plot(mu_theory, yaw_whu, 'o', color=colors['WHU'], 
                       markersize=3, label='WHU', alpha=0.8)
            
            # CSNO模型 (适用于C25)
            if sat == 'C25':
                yaw_csno = [calculate_csno_model_yaw(mu, beta) for mu in mu_theory]
                ax.plot(mu_theory, yaw_csno, 'o', color=colors['CSNO'], 
                       markersize=3, label='CSNO', alpha=0.8)
            
            # 绘制四元数数据 (如果有对应的数据)
            if center_name in ['WHU-RAP', 'WHU-FIN']:
                center_key = center_name.replace('-', '_')
                if center_key in all_data and sat in all_data[center_key]:
                    epochs = all_data[center_key][sat]['epochs']
                    quaternions = all_data[center_key][sat]['quaternions']
                    
                    if epochs and quaternions:
                        mu_angles = []
                        yaw_angles = []
                        
                        for epoch, q in zip(epochs, quaternions):
                            mu = calculate_orbital_angle(epoch)
                            yaw = quaternion_to_yaw(np.array(q))
                            
                            # 只保留-20到20度范围内的数据
                            if -20 <= mu <= 20:
                                mu_angles.append(mu)
                                yaw_angles.append(yaw)
                        
                        if mu_angles:
                            # 排序数据
                            sorted_indices = np.argsort(mu_angles)
                            mu_sorted = np.array(mu_angles)[sorted_indices]
                            yaw_qua_sorted = np.array(yaw_angles)[sorted_indices]
                            
                            # 处理角度跳跃
                            for j in range(1, len(yaw_qua_sorted)):
                                if yaw_qua_sorted[j] - yaw_qua_sorted[j-1] > 180:
                                    yaw_qua_sorted[j:] -= 360
                                elif yaw_qua_sorted[j] - yaw_qua_sorted[j-1] < -180:
                                    yaw_qua_sorted[j:] += 360
                            
                            ax.plot(mu_sorted, yaw_qua_sorted, color=colors['QUA'], 
                                   linewidth=2, label='QUA', alpha=0.8)
            
            # 添加灰色阴影区域 (机动区域)
            ax.axvspan(-20, -10, alpha=0.3, color='gray')
            ax.axvspan(10, 20, alpha=0.3, color='gray')
            
            # 设置图形属性
            ax.set_xlim(-20, 20)
            ax.set_ylim(0, 180)
            ax.set_xlabel('Orbital Angle μ (deg)', fontsize=10)
            ax.set_ylabel(f'{sat_name} Yaw Angle (deg)', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 设置刻度
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_yticks([0, 60, 120, 180])
            
            # 设置标题
            ax.set_title(f'{center_name}', fontsize=11)
            
            # 只在第一个子图添加图例
            if sat_idx == 0 and center_idx == 0:
                ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle('Fig.3 Comparison of quaternion yaw angle, nominal yaw angle and model yaw angle for satellite midnight maneuver', 
                 fontsize=13, y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """主函数"""
    # 数据目录
    data_dir = '/Users/xingqiangchen/TASK/obx/obx-dataset'
    
    print("正在加载子夜机动期间的OBX数据...")
    all_data = load_midnight_data(data_dir)
    
    print("正在生成图3...")
    plot_figure3(all_data, 'figure3_reproduction.png')
    
    print("图3生成完成！")

if __name__ == "__main__":
    main()
