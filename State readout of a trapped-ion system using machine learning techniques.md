# State readout of a trapped-ion system using machine learning techniques



## 前言

我们将研究前馈神经网络用于捕获离子系统状态读出的机制。前馈神经网络作为线性分类器，学习检测器上离子的点扩散函数(PSF)。此外，我们还介绍了我们开发的用于执行状态读出的算法，该算法能够适应离子和不同数量离子的运动。



---



在所有的量子信息处理**QIP（Quantum Information Processing）[^QIP]**实验中，在实验结束时测量状态的能力对于理解结果至关重要。

**目标：**在俘获离子平台上，使用荧光测量来标记离子，发射**共振激光束[^非共振激发]**来将qubit置为激发态，离子会持续发光，我们需要将其与其他不发光的离子区分开来。在多离子设置中，我们需要能够区分每个离子发射的光子。

本文在图像上使用**阈值方法[^阈值方法]**（光子计数）来识别离子的状态。

机器学习的方法**（前馈神经网络FFNN）**能够达到与传统方法（自适应阈值）一样的准确率，甚至更准确。

本文提出了一个确定每个离子感兴趣区域的算法（**区域兴趣ROI[^ROI]**）（可以使用GAM来可视化ROI）





## Detection theory



<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231103141903178.png" alt="image-20231103141903178" style="zoom:80%;" />

​		在测量过程中，使用σ±和π极性的谐振激光器来驱动转换(红色)，将基态的$|S_{1/2},F=1\rang$耦合到$|P_{1/2},F=0,m_F=0\rang$态。（由基态S1/2 到激发态P1/2 ，由稳定到活泼）



将|1>态的离子转换到$|P_{1/2},F=0,m_F=0\rang$态，|1>态的离子经历循环转变而持续发出荧光

然而，对于处于|0 >状态的离子，探测光束通过ωHFS(12.643GHz) + ωHFP(2.105GHz)的因数从它们的允许跃迁到$|P_{1/2},F=1,m_F=0,\pm1\rang$状态，其中ωHFS和ωHFP分别是S1/2和P1/2状态的超细分裂。因此，|0 >和|1 >状态是可区分的。





**亮离子（即在检测光束下发出荧光的处于状态 |1> 的离子）**的光子统计数据将遵循**泊松分布**，具体取决于以下参数：

​		$\gamma$ : 辐射线宽，通常与能级的宽度相关，表示光谱线的宽度。
​		$\eta$ : 探测器的光子收集效率，表示探测器捕捉到的光子占发射的光子的比例。
​		$\delta$ : 由循环跃迁引起的探测光束失谐，表示光谱线与探测器的响应之间的失谐。
​		$I_D$ : 探测光束的强度，即光子的数量。
​		$\tau_D$ : 用于进行光子计数的时间段，通常以秒为单位。

**“未考虑泄漏[^leakage]”**光子计数概率分布有如下形式（在|1>状态下，发射k个光子的概率）：
$$
Pr(X=k;|1\rangle) = Poisson(k;\lambda_0) = \frac{\lambda_{0}^{k}e^{-\lambda_0}}{k!},
$$

$$
\lambda_0 = \tau_{D}\eta\frac{s{\frac{\gamma}{2}}}{1 + s + (\frac{2\delta}{\gamma})^2}.
$$

​		$X$ : 离子发射的光子数量
​		$\lambda_0$ : 平均光子数
​		$s = I_D / I_{sat}$ : 探测光束的饱和参数
​		$I_D$ : 探测光束的强度，表示光束中光子的数量。
​		$I_{sat}$ : 饱和强度，表示当光束强度达到一定水平时，原子或离子不再响应于光的强度变化。



**假设**上述光子数分布在|1>和|0>态之间没有泄漏。（不会相互转换的意思）

**“考虑泄漏”**为了解释非共振激发引起的泄漏，通过将上述泊松分布与指数分布(与状态泄漏对应的概率分布)卷积来修改暗态和亮态的光子计数概率分布。这句话表明在光子计数的概率分布中，除了考虑泊松分布描述的光子计数外，还需要考虑状态泄漏的影响，这是通过**将泊松分布与指数分布卷积**来完成的。这种方法可以更准确地描述实验或观测中可能发生的状态泄漏现象。
$$
\begin{aligned}
\operatorname{Pr}(X=k ;|0\rangle) & =\exp \left(-\frac{\alpha_1 \lambda_0}{\eta}\right)\left[\delta_n+\frac{\frac{\alpha_1}{\eta}}{\left(1-\frac{\alpha_1}{\eta}\right)^{k+1}} \Gamma\left(k+1,\left(1-\frac{\alpha_1}{\eta}\right) \lambda_0\right)\right], \\
\operatorname{Pr}(X=k ;|1\rangle) & =\frac{\lambda_0^k \exp \left(-\left(1+\frac{\alpha_2}{\eta}\right) \lambda_0\right)}{k !}+\frac{\frac{\alpha_2}{\eta}}{\left(1+\frac{\alpha_2}{\eta}\right)^{k+1}} \Gamma\left(k+1,\left(1+\frac{\alpha_2}{\eta}\right) \lambda_0\right), \\
\Gamma(a, b) & =\int_0^b t^{a-1} e^{-t} d t .
\end{aligned}
$$
​		$a_1, a_2$ : |0>、|1>状态下的泄漏概率，取决于原子参数
​		$\Gamma$ : 下不完全伽玛函数
​		$\delta_n$ : 克罗内克函数





171Yb的光子计数分布的一个例子，$a_1 = 1.686\texttimes 10^{-7}, a_2 = 8.264\texttimes 10^{-6}, \lambda_0 = 19.67$：

<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231103164815844.png" alt="image-20231103164815844" style="zoom:70%;" />

​		明暗171Yb离子的光子计数统计。其中，平均亮光子数λ0 = 19.67。在暗态|0 >的图中省略了X = 0光子的概率。在没有泄漏的情况下，我们没有得到状态|0 >的光子（意思是没泄漏的话，|0>本来就不发射光子）。





为了量化状态读出协议的性能，我们使用成功概率（即协议正确分类离子状态的概率）为过程定义保真度 F。基于上述分布，我们获得了状态读出的**理论最大保真度 Fmax**，
$$
F_{max} = \frac{1}{2} \Sigma_{k}{max(Pr(X=k;|0\rang), Pr(X=k;|1\rang))}
$$
这个公式计算了在不同光子计数值 k 下，暗态 |0〉 和亮态 |1〉 的概率中的最大值（with leakage），并将它们相加，然后除以 2。
对于上面的分布，$F_{max} = 0.99969$ 





在上面，有了暗态和亮态发射的光子数的分布。对于多离子系统，必须确定光子的空间分布。假设离子是光子的点源，离子在探测器上有一个**点扩散函数(PSF)[^PSF]**，它描述了离子发射的光子在特定位置撞击探测器的概率。在成像系统有衍射限制的情况下，我们得到了一个遵循**Airy模式[^Airy pattern]**的PSF，
$$
PSF(x,y) = \frac{2J_1(kasin[\theta(x,y)])}{kasin[\theta(x,y)]}
$$
​		$J_1()$ : 一阶贝塞尔函数（Bessel function of the first kind），数学中的特殊函数，与衍射有关。
​		$k$ : 波数（wave number），k通常等于2π除以光的波长。
​		$a$ : 孔径的半径，表示光通过的孔径的大小，这影响了成像的分辨能力。
​		$\theta (x,y)$ : 观测角度，它是入射光线与(x, y)坐标处的位置之间的夹角。这个角度决定了光线的传播方向和传播路径。





**“像素化”**由于探测器具有有限的空间分辨率，因此它们产生的图像是像素化的，即由一系列离散的像素组成。为了考虑这种情况，需要对点扩散函数（PSF）进行离散化处理，结果就是离散化点扩散函数（Discretized Point Spread Function，DPSF[^DPSF]）
$$
\begin{aligned}
\operatorname{DPSF}_{m n} & =\int_{y_n}^{y_{n+1}} \int_{x_m}^{x_{m+1}} \operatorname{PSF}(x, y) d x d y, \\
\alpha_m & =m \Delta \alpha, \text { for } \alpha \in\{x, y\} .
\end{aligned}
$$
- $\Delta \alpha$ 是一个常数，用来表示坐标 $\alpha$ 的离散步长或间隔。
- $\alpha_m$ 表示第 $m$ 个离散步长的坐标值。具体来说，它表示 $x$ 或 $y$ 坐标中的第 $m$ 个步长的位置。





<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231103184039564.png" alt="image-20231103184039564" style="zoom:70%;" />

​		图4.3:离子的点扩散函数(PSF)离子的理论PSF和离子探测器上模拟的DPSF。这里，**Airy半径[^Airy Radius]**是1.95像素。


$$
Pr_{mn}^{noise}(X=k) = Poisson(k;\lambda_{noise}),
$$

$$
\lambda_{noise} = \tau_{D}R_{noise}.
$$

​		$R_{noise}$ : 探测器记录的错误光子的比率，这些光子不是来自离子的发光，而是由于背景或其他噪声源引起的。（意思是噪声发射k个光子的概率，这k个光子并不是被测离子发出的**？？？**）

本节中提供的概率分布用于生成模拟荧光测量数据，以供随后的机器学习工作使用。





## FFNN for state readout

有了图像，现在要识别图像上的离子状态。使用FFNN，得到阈值和像素贡献。



### NN structure

将图像以一维方式输入到输入层

<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231104093756156.png" alt="image-20231104093756156" style="zoom:80%;" />

**输入层：**输入是一张测量图像$X_{mn}$ ，由探测器决定数量单元数量$D_1 \times D_2$ ，$D_1, D_2$ 是探测器的维度。i.e 50x50

**隐藏层：**输入层全连接到隐藏层的N（N个离子）个线性单元，没有偏置biases。

**输出层：**隐藏层全连接到输出层的N个Sigmoid单元。输出为对每个离子是否处于“亮状态”的概率，即$p_k^{pred}$。

**输出层二值化（Binarization）：** 阈值为0.5，大于阈值“亮”，输出$\psi_i^{pred}$ 

**可训练参数（Trainable Parameters）：**权重矩阵$W_{mnj}^{(1)}, W_{ji}^{(2)}$ 和输出层的偏置$b_{i}^{(2)}$ 



### Training parameters

比较预测概率$p_i^{pred}$ 和实际离子状态$\psi_i$ 之间的差异。$\psi_i \in {\{0, 1 \}}$ 标签值

使用Adam算法[^Adam]来优化二进制交叉熵成本函数（即损失函数）（**文章中的有错误**）：
$$
C(p_i^{pred}, \psi_i) = -\frac{1}{N}\sum_k{p_i^{pred} log(\psi_i) + (1-p_i^{pred})log(1-\psi_i)} \\

C(p_i^{pred}, \psi_i) = -\frac{1}{N}\sum_k{\psi_i log(p_i^{pred}) + (1-\psi_i)log(1-p_i^{pred})}
$$

**训练集：**10000
**验证集：**1000
**batch size:** 100
**lr:** 0.001
**epochs:** 100



### Condensed NN

FFNN可以被压缩（为了简化模型？提高效率？减少过拟合风险？可拓展性？），考虑神经网络的输出对输入的依赖性
$$
p_i^{pred} = Sigmoid(\sum_{m,n,j}{W_{jk}^{(2)} W_{mnj}^{(1)} X_{mn}} + b_i^{(2)})
$$
定义压缩权重$W_{mni}^{cond} = \sum_{j}{W_{ji}^{(2)} W_{mnj}^{(1)}}$ （这种线性网络可以将多层转化为一层）
$$
p_i^{pred} = Sigmoid(\sum_{m,n}{W_{mni}^{cond} X_{mn}} + b_i^{(2)})
$$

一种这样的线性分类器是具有一般形式的自适应阈值方法（ATM）:
$$
\psi_i^{ATM} = Heaviside(\sum_{m,n}{W_{mni}^{threshold} X_{mn} - b_i^{threshold}}).
$$

​		Heaviside是**Heaviside阶跃函数**[^阶跃函数]，$W_{mni}^{threshold},b_i^{threshold}$ 用于决定i离子的状态和阈值。



### Results and analysis



#### 单离子系统：

实验参数：亮离子平均发射光子数$\lambda_0 = 19.67$ ，离子点PSF的Airy radius = 1.95px，每个像素的平均错误光子数$\lambda_{noise} = 0.24$ 。

<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231104142544019.png" alt="image-20231104142544019" style="zoom:70%;" />

有误差的情况下，FFNN的C和1-F都增加了。但Cost还是小于没噪声的MFT[^MFT]。由于MFT和FFNN保真度和成本函数的形式不同，导致保真度略有差异。





NN学习DPSF的特征

<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231105135803106.png" alt="image-20231105135803106" style="zoom:80%;" />

(a) FFNN使用压缩权重和偏置在无噪声的情况下训练（与(c)很像）；
(b) FFNN使用压缩权重和偏置在噪声的情况下训练（中间3x3作为离子的region of interest (ROI)）；
(c) 离子检测器上的模拟 DPSF



#### 多离子系统：

假设离子的PSF相同，并由Airy模式指定(公式6)。

两个离子相距4个px的系统

<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231105143736323.png" alt="image-20231105143736323" style="zoom:80%;" />





不仅要考虑到噪声的影响，还需要考虑两个离子DPSF交叠(离子之间的干扰)的影响，N=2。

<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231106081600123.png" alt="image-20231106081600123" style="zoom:70%;" />

绿线：单个离子的保真率随离子间隔的变化
蓝线：两个离子DPSF交叠大小随离子间隔的变化

当px>2时，DPSF交叠对单个离子保真度的影响微不足道了





<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231106085230020.png" alt="image-20231106085230020" style="zoom:70%;" />



单个离子的状态识别准确度（fidelity）在不同离子数量的情况下都非常接近，无论离子数量N如何。因为他们的间距至少是4px(很大)，离子之间的干扰很小。所以即使有多个离子，每个离子的状态仍然可以被准确地识别。





使用FFNN做状态读出的关键：

1. **神经网络充当线性分类器：** 神经网络的工作原理类似于传统的自适应阈值方法
2. **神经网络能够学习PSF或ROI的完整结构：**可以确定离子的位置



由于神经网络在确定囚禁离子系统状态时学习了每个离子PSF的质心，所以神经网络对离子的任何移动和离子数量的变化都很敏感。任何这种变化都需要重新训练神经网络以适应特定系统。

提出一个对这种变化不敏感的方案，两个阶段：

1. **确定每个离子的ROI：** 确定每个离子的兴趣区域（ROI），即探测器上离子的位置。
2. **使用相应的ROI区分每个离子的状态**





## Two-stage state readout protocol



### Region of interest(ROI) detection

已知每个离子ROI约为3x3，离子特征由DPSF给出。将这些信息应用到类似CNN的算法中，可以确定系统中离子的ROI。

**校准图像$X_{mn}^{calib}$：**所有离子都是明亮的



**步骤：**

1. Doppler Cooling：冷却是减小误差的关键步骤，多普勒冷却。
2. 收集散射光子：即在Doppler Cooling[^Doppler Cooling]期间获得离子的图像，该图像被称为**校准图像**。
3. 定位ROI：利用校准图像，确定每个离子的ROI，即离子在图像上的位置。



**实验参数设定：**

N = 10， 轴向陷阱强度为0.2×2π MHz，冷却至轴向模式的平均光子数〈n〉≈1，离子位置的不确定性约为10纳米，相当于在探测器上的0.01像素。





<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231106153323211.png" alt="image-20231106153323211" style="zoom:70%;" />

​		校准图像示例，具有 10 个离子（红色圆圈），呈线性排列[^linear]（为什么要线性排列？？？），平均间隔为 4 像素。这里，λ0 = 127，λnoise = 0.6。





**ROI检测算法流程：**

1. 使用阈值法（阈值1=5）过滤校准图像中的噪声(b)：

$$
\tilde{X}_{m n}^{\text {calib }}= \begin{cases}0 & \text { if } X_{m n}^{\text {calib }}<\theta_1 \\ X_{m n}^{\text {calib }} & \text { otherwise }\end{cases}
$$

2. 将过滤后的图像与离子的3 × 3 DPSF进行卷积(c)
3. 去3 x 3中具有最大值并且大于阈值（阈值2=5）的点作为中心像素得到离子的预测位置(d)



<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231106162132309.png" alt="image-20231106162132309" style="zoom:70%;" />





**实验：**

使用包含10000个校准图像的数据集，每个图像中有10个离子，同时使用了经验确定的阈值参数θ1 = 5和θ2 = 5。

评估结果如下：

- 81.0(4)% 的离子的位置被算法准确预测。
- 2.7(4)% 的离子被算法完全漏掉，未能正确预测其位置。
- 16.3(4)% 的离子的位置被算法误判，即算法的预测位置与真实位置相差一个像素，可能是水平、垂直或对角线方向的误差[^cross-talk]。但不会对离子状态的读出有影响。





### State readout



<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231106170601524.png" alt="image-20231106170601524" style="zoom:70%;" />



1.  首先，将校准图像和测量图像输入到ROI检测算法中。ROI检测算法确定每个离子的感兴趣区域（ROI）。
2. ROI算法确定了每个离子的ROI后，测量图像被分解成每个离子的独立测量图像。
3. 最后，将这些独立测量图像分别输入到状态读出FFNN中，每个离子都有一个对应的状态输出，即$\psi_i^{prrd}$ ，表示每个离子的状态的预测值。



保真度$F_{single} = 0.9915$ ，比上面表中有所提高，因为过滤了与离子状态无关的像素。（有疑问🤔）






## Conclusion

前馈神经网络显示了学习离子的PSF特性的能力。然而，由于前馈神经网络对离子位置的敏感性，它并不是理想的。我们概念化了状态检测的两阶段过程，在执行状态读出之前，首先确定离子感兴趣的区域，从而能够执行状态读出。这消除了重新训练神经网络以适应离子数量的变化和探测器上离子位置的任何变化的需要。（分类，能否将目标检测的方法应用到这里）





## 名词解释



### Detection theory：

[^QIP]: **量子信息处理（Quantum Information Processing QIP）**：这是一种利用量子计算机和量子信息理论来处理信息和执行计算的领域。
[^阈值方法]: **阈值方法：** 一旦获得了图像数据（光子计数），通常需要使用阈值方法来识别每个离子的状态。这意味着将图像中的光子计数与某个阈值进行比较，以确定哪些区域对应于处于激发态的离子，哪些区域对应于其他状态的离子。
[^ROI]: **区域兴趣（ROI）：** 通过在实验冷却阶段生成的校准图像上进行，FFNN然后仅在每个离子的兴趣区域上应用，从而形成了一种可扩展的算法，用于读取困扰离子系统的状态。
[^非共振激发]: **非共振激发（non-resonant excitation）：**是指在一个物理系统中，外部激励的频率或能量与系统的自然频率或能量不匹配的情况。在非共振激发中，外部激励与系统的共振频率不同，因此无法实现能量传递或能级跃迁。
[^Detuning]: **Detuning（失谐）：** 失谐是指激光的频率与能级跃迁的共振频率之间的差异。
[^能级]: **｜S1/2, F = 1〉 子能级：** 这表示某个原子的 S1/2 能级中的子能级，其中 F = 1。在原子和分子物理中，F 通常代表原子或分子的总角动量。
[^多离子系统]: **多离子系统：**在研究多个离子（通常是离子阵列或离子晶体）时，每个离子都可以视为一个点光源，可以发射光子。研究人员想要了解这些发射的光子在探测器上的分布情况，以获得离子的空间信息。
[^leakage]: **leakage（泄漏）：**原子或量子比特的态（state）可能不是完全稳定的，它们可能因为各种因素而发生变化或转移。
[^PSF]:  **Point Spread Function (PSF)：** PSF（点扩散函数）是描述光子从点光源（在这种情况下是离子）发射到探测器上的概率分布函数。它表示了一个光子从点源到达探测器不同位置的可能性。
[^Airy pattern]: **艾里模式（Airy pattern）：**如果成像系统受到**衍射限制**，那么PSF将遵循艾里模式。艾里模式是一种典型的衍射图样，描述了点光源在探测器上的光强分布，这是由于光波经过衍射后的特征所引起的。艾里模式通常以中央亮斑和一系列明暗环的形式呈现。

**衍射限制：**是光学成像系统中的一种物理现象，它限制了系统的分辨率和清晰度

[^DPSF]: **离散化点扩散函数（DPSF）：** 为了适应像素化图像的处理，需要将连续的PSF离散化为一个离散的函数，称为DPSF。DPSF描述了在每个像素位置上接收到光子的概率。
[^Airy Radius]: **Airy Radius：** Airy半径是一个参数，通常以像素（px）为单位，用于描述PSF的形状和尺寸。它是Airy模式的一个关键特征，表示在成像系统中由于衍射而导致的模糊区域的尺寸。



### FFNN for state readout:

[^BCECF]: **二进制交叉熵成本函数（Binary cross-entropy cost function）：** 在神经网络的训练中，通常使用成本函数来衡量预测与实际数据之间的差异。
[^Adam]: **ADAM算法：** ADAM是一种梯度下降优化算法，通常用于训练神经网络。
[^阶跃函数]: **Heaviside阶跃函数：**

<img src="/Users/wenlongxin/Library/Application Support/typora-user-images/image-20231104135207819.png" alt="image-20231104135207819" style="zoom:70%;" />

[^MFT]: **MFT:** 最大保真度阈值（MFT）方法
[^cross-talk]: **交叉干扰（cross-talk）：** 这是指在多个离子之间可能发生的信息或干扰交叉传递的现象。





### Two-stage state readout protocol

[^Doppler cooling]: **Doppler cooling:** 一种用于冷却离子或原子的技术，旨在减缓它们的热运动（离子囚禁）。
[^linear]: **10个离子为什么要按线性排列：**

> 1. 简化控制：一维链结构的离子排列相对较简单，使得控制和操作更加容易。它们更容易受到精确的激光束的照射，以执行量子门操作或读出。
> 2. 理论分析：理论分析通常更容易在一维链结构中进行，因为不需要考虑多个维度的相互作用。这使得研究者能够更容易地理解量子态演化和相互作用。
> 3. 技术实现：实际操控离子的实验室通常会选择一维链结构，因为这种结构在实验技术上更易实现，包括离子阱和激光束的布局。



