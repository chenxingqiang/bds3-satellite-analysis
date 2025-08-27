

### 论文研究：BDS-3 MEO卫星地影期偏航姿态分析与光压模型精化

本文针对北斗三号（BDS-3）中圆地球轨道（MEO）卫星在地影期因太阳光压（SRP）模型适用性降低导致的精密定轨精度退化问题，提出了一种组合光压模型构建方法。该方法通过分析卫星偏航姿态变化、精简ECOMC光压模型参数，并辅以先验模型补偿，显著提升了地影期定轨精度。以下是论文中所有公式的提取、目的及含义解释，结构分为三部分：（1）偏航姿态相关公式；（2）光压模型相关公式；（3）先验光压模型公式。公式解释紧密结合其上下文，并在相关位置嵌入原文图片以增强理解。

------

#### 1. **偏航姿态相关公式**

这部分公式用于描述卫星在地影期的偏航姿态变化规律，主要位于论文第1节“BDS-3 MEO卫星偏航姿态分析”。公式的推导和分析基于卫星轨道角（μ）和太阳高度角（β），用于解释名义姿态与实际姿态的差异及机动控制机制。

* **公式（1）：名义偏航姿态角计算**`ψn=atan2(−tanβ,sinμ)`**目的**：计算卫星的名义偏航姿态角（ψn）。该角度定义了卫星在理想条件下（无地影干扰）的姿态方向，是姿态控制的基础参考。**含义**：atan2(a,b)是四象限反正切函数，确保角度值在 [−180∘,180∘]范围内。β为太阳高度角（太阳相对于卫星轨道平面的角度）。μ为卫星轨道角（卫星在轨道平面内的位置角）。公式表明姿态角受太阳高度角和轨道角的联合影响，当地影期 β≈0∘时，姿态角变化剧烈（如图2所示）。**上下文**：该公式用于对比实际偏航姿态（四元数产品）与名义姿态的差异（见图2），揭示地影期需启动机动模式的原因。![img](https://hunyuan-plugin-private-1258344706.cos.ap-nanjing.myqcloud.com/pdf_youtu/img/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-3-8.png?q-sign-algorithm=sha1&q-ak=AKID372nLgqocp7HZjfQzNcyGOMTN3Xp6FEA&q-sign-time=1756266500%3B2071626500&q-key-time=1756266500%3B2071626500&q-header-list=host&q-url-param-list=&q-signature=2f2a545319afd7e5aec5f426887605aedb9eab33)
* **公式（2）：偏航角速率计算**`ψ˙n=sin2μ+tan2βμ˙tanβcosμ`**目的**：推导名义偏航角速率（ψ˙n），分析姿态控制系统是否需启动机动模式。**含义**：μ˙为平均运动角速率（常量，约 0.008∘/s）。当 β→0∘（地影期），分母 sin2μ+tan2β趋近于0，导致角速率急剧增大，超出控制系统阈值（约 0.1∘/s），此时卫星需切换至连续动偏模式。公式解释了图3中卫星在轨道角 μ=0∘/180∘附近提前启动机动的原因。**上下文**：结合图3的机动过程分析，说明深度地影期姿态变化的物理机制。![img](https://hunyuan-plugin-private-1258344706.cos.ap-nanjing.myqcloud.com/pdf_youtu/img/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-3-16.png?q-sign-algorithm=sha1&q-ak=AKID372nLgqocp7HZjfQzNcyGOMTN3Xp6FEA&q-sign-time=1756266512%3B2071626512&q-key-time=1756266512%3B2071626512&q-header-list=host&q-url-param-list=&q-signature=4227c53e47cfcf88e8f5812563dbe151993bcbad)

------

#### 2. **光压模型相关公式**

位于论文第2节“ECOMC模型参数分析及精化”，这些公式定义了经验型光压模型（ECOMC）及其简化版本。ECOMC模型将太阳光压力分解为三个正交方向（D、Y、B），但因地影期参数相关性高，论文通过参数反演和相关性分析进行了精简。

* **ECOMC原始模型公式**`⎩⎨⎧aDaYaB=D0+Dccos(Δu)+Dssin(Δu)+D2ccos(2Δu)+D2ssin(2Δu)+D4ccos(4Δu)+D4ssin(4Δu)=Y0+Yccos(Δu)+Yssin(Δu)=B0+Bccos(Δu)+Bssin(Δu)`**目的**：计算卫星在D（卫星-太阳方向）、Y（星固系Y轴）、B（右手法则方向）三个方向的光压加速度。**含义**：aD,aY,aB为加速度分量（单位：nm/s²）。Δu为卫星相对太阳的轨道角。D0,Y0,B0为常数项，表示稳态光压摄动。周期项参数（如 Dc,Ds）吸收因卫星受照面积周期性变化引起的摄动误差。因地影期参数强相关性（如 D0与 Bc相关系数 >0.8，见图8），模型精度下降。**上下文**：图6展示ECOMC模型轨道拟合残差在地影期显著增大（尤其是CAST-MEO卫星），验证了精简必要性。![img](https://hunyuan-plugin-private-1258344706.cos.ap-nanjing.myqcloud.com/pdf_youtu/img/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-5-10.png?q-sign-algorithm=sha1&q-ak=AKID372nLgqocp7HZjfQzNcyGOMTN3Xp6FEA&q-sign-time=1756266534%3B2071626534&q-key-time=1756266534%3B2071626534&q-header-list=host&q-url-param-list=&q-signature=2f52f52843550e2356fd38e0263d553f52998788)![img](https://hunyuan-plugin-private-1258344706.cos.ap-nanjing.myqcloud.com/pdf_youtu/img/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-6-17.png?q-sign-algorithm=sha1&q-ak=AKID372nLgqocp7HZjfQzNcyGOMTN3Xp6FEA&q-sign-time=1756266536%3B2071626536&q-key-time=1756266536%3B2071626536&q-header-list=host&q-url-param-list=&q-signature=3c8c0a46314da163a386e6408021e72bca41a37f)
* **公式（4）：简化ECOMC模型**`⎩⎨⎧aD=D0+Dccos(Δu)+D2ccos(2Δu)+D4ccos(4Δu)aY=Y0+Yssin(Δu)aB=B0+Bssin(Δu)`**目的**：精简ECOMC模型参数，提升地影期定轨稳定性。**含义**：通过参数振幅分析（表2），剔除贡献小的参数（如 Ds,Yc,D2s,D4s均方根 <0.2 nm/s²），仅保留主导项。D方向保留余弦项（Dc,D2c,D4c），因正弦项（Ds,D2s,D4s）在地影期趋近于零（图7）。Y和B方向仅保留 sin(Δu)项，因 Ys和 Bs对周期性摄动拟合更有效。参数数量从13个减至8个，降低相关性（如 D0与 Bc相关性消除）。**上下文**：图9对比简化后参数估计更平滑，离散度降低，验证了精简有效性。![img](https://hunyuan-plugin-private-1258344706.cos.ap-nanjing.myqcloud.com/pdf_youtu/img/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-6-5.png?q-sign-algorithm=sha1&q-ak=AKID372nLgqocp7HZjfQzNcyGOMTN3Xp6FEA&q-sign-time=1756266554%3B2071626554&q-key-time=1756266554%3B2071626554&q-header-list=host&q-url-param-list=&q-signature=66c7cf6cfed4d172caa21388a268deea987568d6)![img](https://hunyuan-plugin-private-1258344706.cos.ap-nanjing.myqcloud.com/pdf_youtu/img/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-8-8.png?q-sign-algorithm=sha1&q-ak=AKID372nLgqocp7HZjfQzNcyGOMTN3Xp6FEA&q-sign-time=1756266556%3B2071626556&q-key-time=1756266556%3B2071626556&q-header-list=host&q-url-param-list=&q-signature=44d5145fef0b1e575b8b53ecbcf4daf4ff09c078)

------

#### 3. **先验光压模型公式**

位于第2.3节“组合光压模型构建”，论文引入长方体卫星先验模型补偿简化ECOMC的残余误差。公式（5）–（8）基于卫星几何特征（如表面积、反射系数）建模光压摄动。

* **公式（5）：D方向先验光压加速度**`acube,D=−aCaδ(∣cosε∣+sinε+32)−aSaδ(∣cosε∣−sinε−34sin2ε+32)−aAaδ(32∣cosε∣cosε+cosε)−2aCρ(∣cosε∣cos2ε+sin3ε)−2aSρ(∣cosε∣cos2ε−sin3ε)−2aAρcos3ε`**目的**：计算D方向光压加速度，考虑卫星本体立方体结构（C）、拉伸（S）和对称（A）部分的吸收与反射效应。**含义**：ε为太阳方位角（太阳方向与卫星本体坐标系的夹角）。aCaδ,aSaδ,aAaδ为吸收和漫反射系数相关的加速度分量（由公式（7）定义）。aCρ,aSρ,aAρ为镜面反射系数相关的加速度分量。公式综合了热辐射和潜在辐射体效应，补偿地影期ECOMC模型未涵盖的摄动。
* **公式（6）：B方向先验光压加速度**`acube,B=−34aSaδ(cosεsinε)−32aAaδ(∣cosε∣sinε)−2aCρ(∣cosε∣−sinε)cosεsinε)−2aSρ(∣cosε∣+sinε)cosεsinε)−2aAρcos2εsinε`**目的**：计算B方向光压加速度，侧重拉伸（S）和对称（A）部分的横向摄动贡献。**含义**：结构与公式（5）类似，但针对B方向的力矩平衡设计。用于校正因卫星不对称性引起的法向（C）误差（表1显示B方向误差显著）。
* **辅助公式（7）和（8）**`aCaδ=21(aZaδ+a∗Xaδ),aSaδ=21(aZaδ−a∗Xaδ),aAaδ=21(a∗Zaδ−a∗Xaδ)``aiaδ=mcAiΦ0(αi+δi),aiρ=mcAiΦ0ρi`**目的**：定义先验模型系数，关联卫星物理属性（如表面积 Ai、质量 m）。**含义**：Φ0为太阳常数（1 AU处的光通量）。αi,δi,ρi分别为吸收、漫反射和镜面反射系数（表3给出CAST/SECM卫星实测值）。公式将光压摄动转化为可测物理量，提升模型普适性。**上下文**：组合模型（简化ECOMC + 先验模型）使地影期定轨精度提升（表6），SLR残差降低（表7）。

------

#### 总结

论文中公式系统化描述了卫星偏航姿态动力学（公式1–2）、光压经验模型（ECOMC及简化版公式4）和物理驱动先验模型（公式5–8）。关键贡献在于：

* 偏航姿态公式揭示了地影期机动机制（图3），指导选用WHU/CSNO模型。
* ECOMC精简通过参数相关性分析（图8）和振幅评估（图7），减少参数至8个。
* 先验模型引入几何约束（公式5–8），补偿了残余误差，最终组合模型使CAST-MEO卫星径向精度提升21.22–26.09%，法向精度提升10.87–28.52%（表6）。这些公式共同解决了BDS-3卫星地影期定轨精度退化问题，为GNSS高精度服务提供支撑。





文档中找出所有公式：

1. 1.**公式（1）**：卫星名义偏航姿态角的表达式。`ψn=atan2(−tanβ,sinμ)`目的：计算卫星的名义偏航姿态角。其中，*β*是太阳高度角，*μ*是卫星轨道角。函数 atan2(*a*,*b*)是反正切函数，用于确定角度。

2. 2.**公式（2）**：偏航角速率的表达式。`ψ˙n=sin2μ+tan2βμ˙tanβcosμ`目的：计算偏航角速率，其中 *μ*˙是平均运动角速率。这用于分析当太阳高度角变化时，偏航姿态是否需要机动。

3. 3.**ECOMC 光压模型公式**：在文档中，ECOMC 模型的加速度计算表达式。

4. `目的：描述卫星在 D、Y、B 三个方向上的光压加速度。其中，*a**D*, *a**Y*, *a**B*是加速度分量；*D*0, *Y*0, *B*0是常数项；其他是周期项参数；Δ*u*是卫星相对太阳的轨道角。这用于建模太阳光压摄动。

5. 4.**公式（4）**：简化后的 ECOMC 模型。`⎩⎨⎧aD=D0+Dccos(Δu)+D2ccos(2Δu)+D4ccos(4Δu)aY=Y0+Yssin(Δu)aB=B0+Bssin(Δu)`目的：在精化模型中，通过分析参数相关性，简化了 ECOMC 模型，只保留贡献较大的参数。

6. 5.**公式（5）和（6）**：先验光压模型的加速度计算。•**公式（5）**：`acube,D=−aCaδ(∣cosε∣+sinε+32)−aSaδ(∣cosε∣−sinε−34sin2ε+32)−aAaδ(32∣cosε∣cosε+cosε)−2aCρ(∣cosε∣cos2ε+sin3ε)−2aSρ(∣cosε∣cos2ε−sin3ε)−2aAρcos3ε`目的：计算卫星在 D 方向上的光压摄动加速度，使用先验模型。考虑了卫星本体的立方体形状、拉伸和对称部分的影响。•**公式（6）**：`acube,B=−34aSaδ(cosεsinε)−32aAaδ(∣cosε∣sinε)−2aCρ(∣cosε∣−sinε)cosεsinε)−2aSρ(∣cosε∣+sinε)cosεsinε)−2aAρcos2εsinε`目的：计算在 B 方向上的光压摄动加速度，同样使用先验模型。

7. 6.**公式（7）**：定义中间变量。`aCaδ=21(aZaδ+a∗Xaδ),aSaδ=21(aZaδ−a∗Xaδ),etc.`目的：定义先验模型中的系数，用于计算加速度。这些是辅助定义。

8. 7.**公式（8）**：光压加速度的基本表达式。`aiaδ=mcAiΦ0(αi+δi),aiρ=mcAiΦ0ρi`目的：计算光压加速度的基本公式，其中 *i*表示不同平面；*A**i*是表面积；Φ0是太阳常数；*m*是卫星质量；*c*是光速；*α**i*, *δ**i*, *ρ**i*是吸收、漫反射和镜面反射系数。

   

   ### 基于OBX数据构建深度学习模型的可行性分析与实现方案

   #### 一、论文方案与OBX数据的适配性分析

   论文的核心创新点在于**地影期偏航姿态建模**和**光压模型精化**，而OBX数据集完美匹配论文数据需求：

   1. **数据同源性**：OBX数据来自武汉大学（WHU），与论文中验证的WHU_FIN/WHU_RAP产品一致
   2. **关键要素覆盖**：包含论文重点卫星C23/C25的四元数姿态数据时间窗口（2024年105-112天）覆盖典型地影期（见图1太阳高度角变化）30秒采样率满足姿态变化捕捉需求

   

   ![img](./%E8%AE%BA%E6%96%87%E7%A0%94%E7%A9%B6BDS-3%20MEO%E5%8D%AB%E6%98%9F%E5%9C%B0%E5%BD%B1%E6%9C%9F%E5%81%8F%E8%88%AA%E5%A7%BF%E6%80%81%E5%88%86%E6%9E%90%E4%B8%8E%E5%85%89%E5%8E%8B%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%8C%96.assets/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-3-3.png)

   

   #### 二、深度学习模型构建框架

   基于论文方案，设计三阶段深度学习架构：

   ```
   graph TD
   A[OBX原始数据] --> B[数据预处理模块]
   B --> C[双分支深度学习模型]
   C --> D[输出与验证]
   B -->|四元数转换| B1[偏航姿态修正模型]
   B -->|轨道参数提取| B2[光压补偿模型]
   C --> C1[LSTM-Attention姿态预测]
   C --> C2[图卷积光压建模]
   D --> D1[偏航角误差评估]
   D --> D2[定轨精度验证]
   ```

   ##### 1. 数据预处理模块

   ```
   def preprocess_obx(file):
       # 解析ORBEX文件
       header, satellites = parse_obx(file)  
       
       # 四元数转欧拉角（重点提取偏航角）
       yaw_angles = [quaternion_to_euler(q0,q1,q2,q3)[2] for sat in satellites]  
       
       # 计算辅助参数
       beta = calc_sun_elevation(sat.position, time)  # 太阳高度角
       mu = calc_orbit_angle(sat.position)           # 轨道角
       
       return {
           'time': timestamps,
           'yaw': yaw_angles,
           'beta': beta,
           'mu': mu,
           'sat_type': sat.metadata['type']  # CAST/SECM标识
       }
   ```

   ##### 2. 双分支模型架构

   **分支一：偏航姿态修正模型（LSTM-Attention）**

   ```
   # 输入层
   inputs = Input(shape=(None, 4))  # [t, β, μ, nominal_yaw]
   
   # 时空特征提取
   x = LSTM(128, return_sequences=True)(inputs)
   x = Attention()(x)  # 聚焦地影期关键时段
   
   # 残差学习
   residual = Dense(32)(inputs)
   x = concatenate([x, residual])
   
   # 输出姿态修正量
   outputs = Dense(1)(x)  # Δψ
   ```

   **分支二：光压补偿模型（Graph Convolution）**

   ```
   # 卫星节点特征：CAST/SECM类型标识 + 轨道参数
   node_features = [...]
   
   # 构建卫星空间关系图
   graph = GraphConv(
       filters=64,
       kernel_size=3,
       edges=calc_satellite_adjacency()  # 基于轨道面构建邻接矩阵
   )(node_features)
   
   # 多周期特征融合
   freq = [1, 2, 4]  # 对应ECOMC的Δu, 2Δu, 4Δu
   x = MultiFrequencyConv(freq)(graph)
   
   # 输出光压补偿值
   outputs = Dense(3)(x)  # [Δa_D, Δa_Y, Δa_B]
   ```

   ##### 3. 损失函数设计

   ```
   # 姿态分支损失
   yaw_loss = cosine_distance(yaw_true, yaw_pred)
   
   # 光压分支损失
   srp_loss = composite_loss(
       physics_constraint = 0.3 * radiation_pressure_constraint(),
       data_fitting = 0.7 * huber_loss(a_true, a_pred)
   )
   
   total_loss = 0.6*yaw_loss + 0.4*srp_loss
   ```

   #### 三、关键技术创新点

   1. **物理引导的深度学习**嵌入公式(1)(2)的偏航动力学约束`ψ˙n=sin2μ+tan2βμ˙tanβcosμ`引入ECOMC模型结构先验（公式3）`aD=D0+Dccos(Δu)+...`
   2. **地影期注意力机制**`class EclipseAttention(Layer):    def call(self, inputs):        # 增强|β|<12.9°区域的权重        weights = tf.where(abs(beta)<12.9, 3.0, 1.0)        return inputs * weights`
   3. **多卫星协同学习**利用图卷积捕获CAST/SECM卫星群的空间关联实现参数共享与知识迁移

   #### 四、预期效果与验证方案

   | 评估指标       | ECOMC基准 | 论文方案 | 深度学习模型预期 |
   | -------------- | --------- | -------- | ---------------- |
   | 径向误差(cm)   | 8.01      | 6.31     | ≤5.80            |
   | 法向误差(cm)   | 4.60      | 4.10     | ≤3.90            |
   | SLR残差(cm)    | 7.31      | 5.52     | ≤5.20            |
   | 日边界跳变(cm) | 9.45      | 7.74     | ≤7.20            |

   **验证方法**：

   1. **偏航角可视化对比**（复现图2效果）`plt.plot(model_pred, label='DL Prediction') plt.plot(obx_data['yaw'], '--', label='OBX Ground Truth') plt.plot(nominal_yaw, ':', label='Nominal')`

   

   ![img](./%E8%AE%BA%E6%96%87%E7%A0%94%E7%A9%B6BDS-3%20MEO%E5%8D%AB%E6%98%9F%E5%9C%B0%E5%BD%B1%E6%9C%9F%E5%81%8F%E8%88%AA%E5%A7%BF%E6%80%81%E5%88%86%E6%9E%90%E4%B8%8E%E5%85%89%E5%8E%8B%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%8C%96.assets/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-3-8.png)

   

   1. **光压参数分析**（复现图7效果）`# 反演D0参数变化曲线 inverse_ecomc(model_output)`

   

   ![img](./%E8%AE%BA%E6%96%87%E7%A0%94%E7%A9%B6BDS-3%20MEO%E5%8D%AB%E6%98%9F%E5%9C%B0%E5%BD%B1%E6%9C%9F%E5%81%8F%E8%88%AA%E5%A7%BF%E6%80%81%E5%88%86%E6%9E%90%E4%B8%8E%E5%85%89%E5%8E%8B%E6%A8%A1%E5%9E%8B%E7%B2%BE%E5%8C%96.assets/hy-20250827113017_629aa0bcb619c45734b622830d445aaa-6-5.png)

   

   #### 五、实施挑战与解决方案

   1. **数据局限**：问题：仅有8天OBX数据方案：采用Meta-Learning（MAML算法）实现小样本学习
   2. **物理约束整合**：`# 微分方程约束层 class DynamicsConstraint(Layer):    def call(self, inputs):        # 实现公式(2)的偏航角速率约束        dyaw_dt = calc_yaw_rate(beta, mu)        return tf.reduce_mean(tf.square(output - dyaw_dt))`
   3. **实时性要求**：轻量化模型：Depthwise Separable Convolution量化压缩：FP16精度 + TensorRT部署

   #### 六、结论

   基于OBX数据构建深度学习模型完全可行且具有显著优势：

   1. **数据适配性**：OBX包含论文分析所需的核心要素（四元数姿态、时间窗口等）
   2. **性能预期**：通过双分支架构和物理引导设计，有望超越论文中21.22%的精度提升
   3. **扩展价值**：模型可迁移至其他GNSS系统（GPS/Galileo）

   > **实施建议**：优先在C23/C25卫星上验证，逐步扩展至全星座。重点优化地影期（|β|<12.9°）的预测精度，该时段对最终定轨精度影响占比达68%（论文图6数据）。