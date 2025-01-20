import numpy as np
import matplotlib.pyplot as plt


def wind_turbine_cp(lambda_values, pitch_angle):
    """
    计算风力发电机的功率系数 Cp

    参数:
    lambda_values: 叶尖速比数组
    pitch_angle: 桨距角 (度)
    """
    c1 = 0.5176
    c2 = 116
    c3 = 0.4
    c4 = 5
    c5 = 21
    c6 = 0.0068

    theta = pitch_angle * np.pi / 180  # 转换为弧度

    a = 1 / (1 / (lambda_values + 0.08 * theta) - 0.035 / (theta ** 3 + 1))
    cp = c1 * (c2 / a - c3 * theta - c4) * np.exp(-c5 / a) + c6 * lambda_values

    return np.clip(cp, 0, 0.59)  # Betz限制


# 设置参数
lambda_range = np.linspace(0, 15, 1000)
pitch_angles = [0, 5, 10, 15]

# 绘图
plt.figure(figsize=(10, 6))

for pitch in pitch_angles:
    cp = wind_turbine_cp(lambda_range, pitch)
    plt.plot(lambda_range, cp, label=f'Pitch = {pitch}°')

plt.xlabel('Tip Speed Ratio (λ)')
plt.ylabel('Power Coefficient (Cp)')
plt.title('Wind Turbine Power Coefficient vs Tip Speed Ratio')
plt.legend()
plt.grid(True)

# 添加Betz限制线
plt.axhline(y=16 / 27, color='r', linestyle='--', label='Betz Limit')

plt.ylim(0, 0.6)
plt.legend()
plt.tight_layout()
plt.show()

# 计算最佳叶尖速比和最大功率系数
best_lambda = lambda_range[np.argmax(wind_turbine_cp(lambda_range, 0))]
max_cp = np.max(wind_turbine_cp(lambda_range, 0))

print(f"最佳叶尖速比: {best_lambda:.2f}")
print(f"最大功率系数: {max_cp:.4f}")


# 计算风力发电机功率
def wind_power(v, r, cp, rho=1.225):
    """
    计算风力发电机功率

    参数:
    v: 风速 (m/s)
    r: 风轮半径 (m)
    cp: 功率系数
    rho: 空气密度 (kg/m^3)
    """
    return 0.5 * rho * np.pi * r ** 2 * v ** 3 * cp


# 计算不同风速下的功率
wind_speeds = np.linspace(0, 25, 100)
power = wind_power(wind_speeds, r=30, cp=max_cp)

plt.figure(figsize=(10, 6))
plt.plot(wind_speeds, power / 1000)  # 转换为kW
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power Output (kW)')
plt.title('Wind Turbine Power Output vs Wind Speed')
plt.grid(True)
plt.tight_layout()
plt.show()