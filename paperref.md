论文研究：BDS-3 MEO卫星地影期偏航姿态分析与光压模型精化

本文针对北斗三号（BDS-3）中圆地球轨道（MEO）卫星在地影期因太阳光压（SRP）模型适用性降低导致的精密定轨精度退化问题，提出了一种组合光压模型构建方法。该方法通过分析卫星偏航姿态变化、精简ECOMC光压模型参数，并辅以先验模型补偿，显著提升了地影期定轨精度。以下是论文中所有公式的提取、目的及含义解释，结构分为三部分：（1）偏航姿态相关公式；（2）光压模型相关公式；（3）先验光压模型公式。公式解释紧密结合其上下文，并在相关位置嵌入原文图片以增强理解。

1. 偏航姿态相关公式

这部分公式用于描述卫星在地影期的偏航姿态变化规律，主要位于论文第1节“BDS-3 MEO卫星偏航姿态分析”。公式的推导和分析基于卫星轨道角（\mu）和太阳高度角（\beta），用于解释名义姿态与实际姿态的差异及机动控制机制。

• 公式（1）：名义偏航姿态角计算  

  \[
  \psi_{n} = \operatorname{atan}2(-\tan\beta, \sin\mu)
  \]
  • 目的：计算卫星的名义偏航姿态角（\psi_n）。该角度定义了卫星在理想条件下（无地影干扰）的姿态方向，是姿态控制的基础参考。  

  • 含义：  

    ◦ \(\operatorname{atan}2(a,b)\) 是四象限反正切函数，确保角度值在 [-180^\circ, 180^\circ] 范围内。  

    ◦ \beta 为太阳高度角（太阳相对于卫星轨道平面的角度）。  

    ◦ \mu 为卫星轨道角（卫星在轨道平面内的位置角）。  

    ◦ 公式表明姿态角受太阳高度角和轨道角的联合影响，当地影期 \beta \approx 0^\circ 时，姿态角变化剧烈（如图2所示）。  

  • 上下文：该公式用于对比实际偏航姿态（四元数产品）与名义姿态的差异（见图2），揭示地影期需启动机动模式的原因。

  

• 公式（2）：偏航角速率计算  

  \[
  \dot{\psi}_{n} = \frac{\dot{\mu} \tan\beta \cos\mu}{\sin^{2}\mu + \tan^{2}\beta}
  \]
  • 目的：推导名义偏航角速率（\dot{\psi}_n），分析姿态控制系统是否需启动机动模式。  

  • 含义：  

    ◦ \dot{\mu} 为平均运动角速率（常量，约 0.008^\circ/s）。  

    ◦ 当 \beta \to 0^\circ（地影期），分母 \sin^2\mu + \tan^2\beta 趋近于0，导致角速率急剧增大，超出控制系统阈值（约 0.1^\circ/s），此时卫星需切换至连续动偏模式。  

    ◦ 公式解释了图3中卫星在轨道角 \mu = 0^\circ/180^\circ 附近提前启动机动的原因。  

  • 上下文：结合图3的机动过程分析，说明深度地影期姿态变化的物理机制。

  

2. 光压模型相关公式

位于论文第2节“ECOMC模型参数分析及精化”，这些公式定义了经验型光压模型（ECOMC）及其简化版本。ECOMC模型将太阳光压力分解为三个正交方向（D、Y、B），但因地影期参数相关性高，论文通过参数反演和相关性分析进行了精简。

• ECOMC原始模型公式  

  \[
  \left\{
  \begin{aligned}
  a_D &= D_0 + D_c \cos(\Delta u) + D_s \sin(\Delta u) + D_{2c} \cos(2\Delta u) + D_{2s} \sin(2\Delta u) + D_{4c} \cos(4\Delta u) + D_{4s} \sin(4\Delta u) \\
  a_Y &= Y_0 + Y_c \cos(\Delta u) + Y_s \sin(\Delta u) \\
  a_B &= B_0 + B_c \cos(\Delta u) + B_s \sin(\Delta u)
  \end{aligned}
  \right.
  \]
  • 目的：计算卫星在D（卫星-太阳方向）、Y（星固系Y轴）、B（右手法则方向）三个方向的光压加速度。  

  • 含义：  

    ◦ a_D, a_Y, a_B 为加速度分量（单位：nm/s²）。  

    ◦ \Delta u 为卫星相对太阳的轨道角。  

    ◦ D_0, Y_0, B_0 为常数项，表示稳态光压摄动。  

    ◦ 周期项参数（如 D_c, D_s）吸收因卫星受照面积周期性变化引起的摄动误差。  

    ◦ 因地影期参数强相关性（如 D_0 与 B_c 相关系数 >0.8，见图8），模型精度下降。  

  • 上下文：图6展示ECOMC模型轨道拟合残差在地影期显著增大（尤其是CAST-MEO卫星），验证了精简必要性。

  
  

• 公式（4）：简化ECOMC模型  

  \[
  \begin{cases}
  a_{D} = D_{0} + D_{c} \cos(\Delta u) + D_{2c} \cos(2\Delta u) + D_{4c} \cos(4\Delta u) \\
  a_{Y} = Y_{0} + Y_{s} \sin(\Delta u) \\
  a_{B} = B_{0} + B_{s} \sin(\Delta u)
  \end{cases}
  \]
  • 目的：精简ECOMC模型参数，提升地影期定轨稳定性。  

  • 含义：  

    ◦ 通过参数振幅分析（表2），剔除贡献小的参数（如 D_s, Y_c, D_{2s}, D_{4s} 均方根 <0.2 nm/s²），仅保留主导项。  

    ◦ D方向保留余弦项（D_c, D_{2c}, D_{4c}），因正弦项（D_s, D_{2s}, D_{4s}）在地影期趋近于零（图7）。  

    ◦ Y和B方向仅保留 \(\sin(\Delta u)\) 项，因 Y_s 和 B_s 对周期性摄动拟合更有效。  

    ◦ 参数数量从13个减至8个，降低相关性（如 D_0 与 B_c 相关性消除）。  

  • 上下文：图9对比简化后参数估计更平滑，离散度降低，验证了精简有效性。

  
  

3. 先验光压模型公式

位于第2.3节“组合光压模型构建”，论文引入长方体卫星先验模型补偿简化ECOMC的残余误差。公式（5）–（8）基于卫星几何特征（如表面积、反射系数）建模光压摄动。

• 公式（5）：D方向先验光压加速度  

  \[
  a_{cube,D} = -a_{C}^{a\delta}(\cos\varepsilon + \sin\varepsilon + \frac{2}{3}) - a_{S}^{a\delta}( \cos\varepsilon - \sin\varepsilon - \frac{4}{3}\sin^{2}\varepsilon + \frac{2}{3}) - a_{A}^{a\delta}(\frac{2}{3} \cos\varepsilon \cos\varepsilon + \cos\varepsilon) - 2a_{C}^{\rho}( \cos\varepsilon \cos^{2}\varepsilon + \sin^{3}\varepsilon) - 2a_{S}^{\rho}( \cos\varepsilon
\cos^{2}\varepsilon - \sin^{3}\varepsilon) - 2a_{A}^{\rho}\cos^{3}\varepsilon
  \]
  • 目的：计算D方向光压加速度，考虑卫星本体立方体结构（C）、拉伸（S）和对称（A）部分的吸收与反射效应。  

  • 含义：  

    ◦ \varepsilon 为太阳方位角（太阳方向与卫星本体坐标系的夹角）。  

    ◦ a_{C}^{a\delta}, a_{S}^{a\delta}, a_{A}^{a\delta} 为吸收和漫反射系数相关的加速度分量（由公式（7）定义）。  

    ◦ a_{C}^{\rho}, a_{S}^{\rho}, a_{A}^{\rho} 为镜面反射系数相关的加速度分量。  

    ◦ 公式综合了热辐射和潜在辐射体效应，补偿地影期ECOMC模型未涵盖的摄动。  

• 公式（6）：B方向先验光压加速度  

  \[
  a_{cube,B} = -\frac{4}{3}a_{S}^{a\delta}(\cos\varepsilon\sin\varepsilon) - \frac{2}{3}a_{A}^{a\delta}(\cos\varepsilon \sin\varepsilon) - 2a_{C}^{\rho}( \cos\varepsilon - \sin\varepsilon)\cos\varepsilon\sin\varepsilon) - 2a_{S}^{\rho}( \cos\varepsilon
 + \sin\varepsilon)\cos\varepsilon\sin\varepsilon) - 2a_{A}^{\rho}\cos^{2}\varepsilon\sin\varepsilon
  \]
  • 目的：计算B方向光压加速度，侧重拉伸（S）和对称（A）部分的横向摄动贡献。  

  • 含义：  

    ◦ 结构与公式（5）类似，但针对B方向的力矩平衡设计。  

    ◦ 用于校正因卫星不对称性引起的法向（C）误差（表1显示B方向误差显著）。  

• 辅助公式（7）和（8）  

  \[
  a_{C}^{a\delta} = \frac{1}{2}(a_{Z}^{a\delta} + a_{X}^{a\delta}), \quad a_{S}^{a\delta} = \frac{1}{2}(a_{Z}^{a\delta} - a_{X}^{a\delta}), \quad a_{A}^{a\delta} = \frac{1}{2}(a_{Z}^{a\delta} - a_{X}^{a\delta})
  \]
  \[
  a_{i}^{a\delta} = \frac{A_{i} \Phi_{0}}{m c} (\alpha_{i} + \delta_{i}), \quad a_{i}^{\rho} = \frac{A_{i} \Phi_{0}}{m c} \rho_{i}
  \]
  • 目的：定义先验模型系数，关联卫星物理属性（如表面积 A_i、质量 m）。  

  • 含义：  

    ◦ \Phi_0 为太阳常数（1 AU处的光通量）。  

    ◦ \alpha_i, \delta_i, \rho_i 分别为吸收、漫反射和镜面反射系数（表3给出CAST/SECM卫星实测值）。  

    ◦ 公式将光压摄动转化为可测物理量，提升模型普适性。  

  • 上下文：组合模型（简化ECOMC + 先验模型）使地影期定轨精度提升（表6），SLR残差降低（表7）。

总结

论文中公式系统化描述了卫星偏航姿态动力学（公式1–2）、光压经验模型（ECOMC及简化版公式4）和物理驱动先验模型（公式5–8）。关键贡献在于：
• 偏航姿态公式揭示了地影期机动机制（图3），指导选用WHU/CSNO模型。

• ECOMC精简通过参数相关性分析（图8）和振幅评估（图7），减少参数至8个。

• 先验模型引入几何约束（公式5–8），补偿了残余误差，最终组合模型使CAST-MEO卫星径向精度提升21.22–26.09%，法向精度提升10.87–28.52%（表6）。

这些公式共同解决了BDS-3卫星地影期定轨精度退化问题，为GNSS高精度服务提供支撑。