import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize

plt.rcParams["font.family"] = "JetBrainsMono NF"
plt.rcParams["font.size"] = 24

# 读取CSV文件
df = pd.read_csv("time.csv")

# 单独提取 'total' 行用于加速比计算
total_row = df[df["Category\\Core Num"] == "total"].iloc[0, 1:].astype(float)

# 移除不需要的行 ('total', 'save', 'plot', 'verbose')
df = df[~df["Category\\Core Num"].isin(["total", "save", "plot", "verbose"])]

# 对调 'energy_history' 和 'crack_phase_solve' 的位置
energy_history_row = df[df["Category\\Core Num"] == "energy_history"]
crack_phase_solve_row = df[df["Category\\Core Num"] == "crack_phase_solve"]
df.loc[energy_history_row.index, :] = crack_phase_solve_row.values
df.loc[crack_phase_solve_row.index, :] = energy_history_row.values

# 提取核心数和各个分类数据
core_nums = df.columns[1:].astype(int)
categories = df["Category\\Core Num"]
data = df.iloc[:, 1:]

# 计算加速比 (Speedup)
speedup = total_row.iloc[0] / total_row


# 首尾点
x0, y0 = core_nums[0], speedup[0]
x_end, y_end = core_nums[-1], speedup[-1]


# 定义二次函数模型
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


# 通过首尾点建立方程
def constraints(x0, y0, x_end, y_end):
    # 通过首尾点的方程
    # f(x0) = a * x0^2 + b * x0 + c = y0
    # f(x_end) = a * x_end^2 + b * x_end + c = y_end
    return np.array([[x0**2, x0, 1], [x_end**2, x_end, 1]])


# 目标函数
def objective(params):
    a, b, c = params
    y_fit = quadratic(core_nums, a, b, c)
    return np.sum((speedup - y_fit) ** 2)


# 通过首尾点得到方程系数
A = constraints(x0, y0, x_end, y_end)
y_constraints = np.array([y0, y_end])


def fit_function(params):
    a, b, c = params
    return np.dot(A, np.array([a, b, c])) - y_constraints


# 使用最小化方法来拟合
initial_guess = [1, 1, 1]  # 初始猜测值
result = minimize(
    lambda params: objective(params),
    initial_guess,
    constraints={"type": "eq", "fun": fit_function},
)

# 提取拟合结果
a, b, c = result.x

# 绘制拟合曲线
x_new = np.linspace(core_nums.min(), core_nums.max(), 100)
speedup_new = quadratic(x_new, a, b, c)


# 对每个类别的数据进行平滑处理
smoothed_data = []
for i in range(data.shape[0]):
    spl_category = make_interp_spline(
        core_nums, data.iloc[i, :], k=3
    )  # 使用三次样条插值
    smoothed_data.append(spl_category(x_new))

# 绘制堆积图
fig, ax1 = plt.subplots(figsize=(16, 9))
colors = ["#8BBDE0", "#F9A490", "#B1CAA2", "#E2CCFF", "#F3E4CF"]
ax1.stackplot(x_new, smoothed_data, labels=categories, colors=colors)

# 添加分界线
ax1.axvline(x=8, color="#f57f7f", linestyle="--", linewidth=2)
ax1.axvline(x=16, color="#f8ae81", linestyle="--", linewidth=2)
ax1.axvline(x=24, color="#85a8ff", linestyle="--", linewidth=2)

# 添加图例
ax1.legend(loc="upper left", frameon=True, title="Categories", ncol=2, fontsize=18)

# 设置第一个y轴标题和标签
ax1.set_xlabel("Core Number")
ax1.set_ylabel("Time (s)")
ax1.set_xlim(1, 32)
ax1.set_xticks([1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32])
ax1.set_yticks(np.linspace(0, 20, 5))

# 在指定区域添加文本
ax1.text(
    4.5,
    np.max(smoothed_data) * 0.9,
    "Performance\nCores",
    color="black",
    ha="center",
    va="center",
)
ax1.text(
    12.5,
    np.max(smoothed_data) * 0.75,
    "Efficient\nCores",
    color="black",
    ha="center",
    va="center",
)
ax1.text(
    20,
    np.max(smoothed_data) * 0.6,
    "Hyper\nThreading",
    color="black",
    ha="center",
    va="center",
)
ax1.text(
    28,
    -2,
    r"$^*$13th Gen Intel$^{\mathrm{R}}$ Core$^{\mathrm{TM}}$ i7-13790F",
    fontsize=14,
    color="black",
    ha="center",
    va="center",
)

# 创建第二个y轴
ax2 = ax1.twinx()
ax2.plot(
    x_new,
    speedup_new,
    color="#9799b2",
    linestyle="-",
    linewidth=2.5,
    label="Speedup Trend",
)

# 设置第二个y轴标题和标签
ax2.set_ylabel("Speedup")
ax2.legend(loc="upper right", frameon=True, fontsize=18)
ax2.set_ylim(0, 4)
ax2.set_yticks([1, 2, 3])

# y = 1
ax2.axhline(y=1, color="#00aac1", linestyle="-.", linewidth=3)

# 显示图表
plt.tight_layout()
plt.show()
