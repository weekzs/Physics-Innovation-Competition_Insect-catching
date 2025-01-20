import numpy as np
import matplotlib.pyplot as plt


def solar_cell_iv_curve(Isc, Voc, n, T):
    """
    生成太阳能电池的I-V曲线

    参数:
    Isc: 短路电流
    Voc: 开路电压
    n: 理想因子
    T: 温度 (K)
    """
    k = 1.380649e-23  # 玻尔兹曼常数
    q = 1.60217663e-19  # 电子电荷

    V = np.linspace(0, Voc, 1000)
    I = Isc - Isc * (np.exp(q * V / (n * k * T)) - 1)
    P = V * I

    return V, I, P


# 设置太阳能电池参数
Isc = 6.0  # 短路电流 (A)
Voc = 0.65  # 开路电压 (V)
n = 1.2  # 理想因子
T = 298  # 温度 (K)

V, I, P = solar_cell_iv_curve(Isc, Voc, n, T)

# 找到最大功率点
max_power_index = np.argmax(P)
Vmp = V[max_power_index]
Imp = I[max_power_index]
Pmax = P[max_power_index]

# 计算填充因子和效率
FF = Pmax / (Isc * Voc)
eta = Pmax / (1000 * 0.1)  # 假设入射光强为1000 W/m^2，电池面积为0.1 m^2

# 绘图
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.plot(V, I, 'b-', label='I-V Curve')
ax2.plot(V, P, 'r-', label='P-V Curve')

ax1.set_xlabel('Voltage (V)')
ax1.set_ylabel('Current (A)', color='b')
ax2.set_ylabel('Power (W)', color='r')

ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')

ax1.annotate(f'MPP\n({Vmp:.2f}V, {Imp:.2f}A)', xy=(Vmp, Imp), xytext=(Vmp - 0.1, Imp + 1),
             arrowprops=dict(arrowstyle='->'))

plt.title('Solar Cell I-V and P-V Characteristics')
plt.grid(True)
plt.legend(loc='upper right')

plt.text(0.05, 0.05, f'Isc = {Isc:.2f} A\nVoc = {Voc:.2f} V\nPmax = {Pmax:.2f} W\nFF = {FF:.2%}\nη = {eta:.2%}',
         transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()