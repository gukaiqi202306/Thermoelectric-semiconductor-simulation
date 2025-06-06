import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import argparse
import sys
import os

# 设置matplotlib中文显示
try:
    # 尝试使用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 设置中文字体失败，图表中的中文可能无法正确显示")

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='P型和N型热电材料效率计算')
    parser.add_argument('--p_type', type=str, required=True, help='P型材料组分 (0.01, 0.02, 0.03)')
    parser.add_argument('--n_type', type=str, required=True, help='N型材料组分 (0.0004, 0.0012, 0.0020)')
    parser.add_argument('--area_ratio', type=float, required=True, help='N型和P型的横截面积比(N/P)')
    parser.add_argument('--Tc', type=float, default=300, help='冷端温度 (K)')
    parser.add_argument('--Th', type=float, default=500, help='热端温度 (K)')
    parser.add_argument('--output_dir', type=str, default="efficiency_results", help='输出目录')
    return parser.parse_args()

# 主要处理流程
args = parse_args()

# 获取参数
value_p = args.p_type
value_n = args.n_type
area_ratio = args.area_ratio
Tc = args.Tc
Th = args.Th

# 创建输出目录
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print(f'使用P型材料参杂率: {value_p}')
print(f'使用N型材料参杂率: {value_n}')
print(f'使用N/P面积比: {area_ratio}')
print(f'温度范围: Tc={Tc}K, Th={Th}K')

# 根据参数选择材料数据文件
if value_p=='0.01':
    a1 = pd.read_excel('P_yuanshi_2_5.xls', header=None).values
elif value_p=='0.02':
    a1= pd.read_excel('P_yuanshi_3_1.xls', header=None).values
elif value_p=='0.03':
    a1 = pd.read_excel('P_yuanshi_3_7.xls', header=None).values
else:
    print(f"错误: 不支持的P型材料组分 {value_p}")
    sys.exit(1)

if value_n=='0.0004':
    a2 = pd.read_excel('N_yuanshi_0.0004.xls', header=None).values
elif value_n=='0.0012':
    a2= pd.read_excel('N_yuanshi_0.0012.xls', header=None).values
elif value_n=='0.0020':
    a2= pd.read_excel('N_yuanshi_0.0020.xls', header=None).values
else:
    print(f"错误: 不支持的N型材料组分 {value_n}")
    sys.exit(1)

# 定义插值函数
def material_P(T):
    sb = CubicSpline(a1[:, 0], a1[:, 1])(T) * 1e-6  # 塞贝克系数
    res = CubicSpline(a1[:, 2], a1[:, 3])(T) * 1e-3  # 电导率
    ZT = CubicSpline(a1[:, 4], a1[:, 5])(T)  # 优值系数
    th = (T * sb ** 2) / (ZT * res)  # 热导率
    return sb, res, th
def material_N(T):
    sb = CubicSpline(a2[:, 0], a2[:, 1])(T) * 1e-6  # 塞贝克系数
    res = CubicSpline(a2[:, 2], a2[:, 3])(T) * 1e-3  # 电导率
    th = CubicSpline(a2[:, 4], a2[:, 5])(T) / 100  # 热导率
    return sb, res, th

def temperature_distribution_P(n, J, Tc, Th, max_iter):  # max_iter是迭代次数
    # 参数初始化
    l = 1
    dx = l / (n - 1)
    T = np.linspace(Tc, Th, n)
    # 迭代求解
    for _ in range(max_iter):
        A = np.zeros((n, n))  # A代表的是系数矩阵，b代表的是AX=b中有边的姐，我们要求的是X
        b = np.zeros(n)
        sb, res, th = material_P(T)
        c1 = J * sb / th
        c2 = -1 / th
        c3 = sb ** 2 * J ** 2 / th
        c4 = -J * sb / th
        c5 = res * J ** 2
        # 边界条件
        A[0, 0] = 1
        b[0] = Tc
        A[-1, -1] = 1
        b[-1] = Th
        # 构造系数矩阵
        for i in range(1, n - 1):
            A[i, i - 1] = 1 / (c2[i] * dx)
            A[i, i] = c4[i + 1] / c2[i + 1] - 1 / (c2[i + 1] * dx) - (1 - c1[i] * dx) / (c2[i] * dx)
            A[i, i + 1] = (1 - c1[i + 1] * dx) / (c2[i + 1] * dx) - c3[i + 1] * dx - (1 - c1[i + 1] * dx) * c4[i + 1] / c2[i + 1]
            b[i] = c5[i - 1] * dx
        try:
            T_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("线性方程组求解失败，请检查系数矩阵。")
            return None
        T = T_new.copy()
    return T

def temperature_distribution_N(n, J, Tc, Th, max_iter):  # max_iter是迭代次数
    # 参数初始化
    l = 1
    dx = l / (n - 1)
    T = np.linspace(Tc, Th, n)
    # 迭代求解
    for _ in range(max_iter):
        A = np.zeros((n, n))  # A代表的是系数矩阵，b代表的是AX=b中有边的姐，我们要求的是X
        b = np.zeros(n)
        sb, res, th = material_N(T)
        c1 = J * sb / th
        c2 = -1 / th
        c3 = sb ** 2 * J ** 2 / th
        c4 = -J * sb / th
        c5 = res * J ** 2
        # 边界条件
        A[0, 0] = 1
        b[0] = Tc
        A[-1, -1] = 1
        b[-1] = Th
        # 构造系数矩阵
        for i in range(1, n - 1):
            A[i, i - 1] = 1 / (c2[i] * dx)
            A[i, i] = c4[i + 1] / c2[i + 1] - 1 / (c2[i + 1] * dx) - (1 - c1[i] * dx) / (c2[i] * dx)
            A[i, i + 1] = (1 - c1[i + 1] * dx) / (c2[i + 1] * dx) - c3[i + 1] * dx - (1 - c1[i + 1] * dx) * c4[i + 1] / c2[i + 1]
            b[i] = c5[i - 1] * dx
        try:
            T_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("线性方程组求解失败，请检查系数矩阵。")
            return None
        T = T_new.copy()
    return T

# # 调用函数并打印结果
# result_P = temperature_distribution_P(n=10, J=-1.5, Tc=300, Th=500, max_iter=10)
# if result_P is not None:
#     print(result_P)
#
# result_N = temperature_distribution_N(n=10, J=25, Tc=300, Th=500, max_iter=10)
# if result_N is not None:
#     print(result_N)


def calculate_efficiency_P(Tc, Th, n, l):
    eff_list_P = []
    J_list_P = []
    dx = l / (n - 1)
    q_P_list=[]
    for j in range(0, 31, 1):
        J = -j
        T = temperature_distribution_P(n, J, Tc, Th, 10)
        sb, res, th = material_P(T)
        c1 = J * sb / th
        c2 = -1 / th
        c3 = sb ** 2 * J ** 2 / th
        c4 = -J * sb / th
        c5 = res * J ** 2

        # 计算热流密度, 定义为 q_P表示P的热流密度
        q_P = np.zeros(n)
        for k in range(1, n):
            q_P[k] = ((1 / dx - c1[k]) * T[k] - T[k - 1] / dx) / (c2[k])
        q_P[0] = (1 - c4[1] * dx) * q_P[1] - c3[1] * dx * T[1] - c5[1] * dx
        # 将每次的q_P(L)给存起来
        q_P_list.append(q_P[-1])
        # 第一个积分
        Cumulative_scoring1 = 0
        # 第二个积分
        Cumulative_scoring2 = 0
        for m in range(1, n):
            T1 = T[m]
            T2 = T[m - 1]
            Cumulative_scoring1 += (sb[m] + sb[m - 1]) / 2 * (T1 - T2)
            Cumulative_scoring2 += (res[m] + res[m - 1]) / 2 * dx

        eff = J * (Cumulative_scoring1 + J * Cumulative_scoring2) / q_P[n - 1]
        eff_list_P.append(eff)
        J_list_P.append(J)

    return eff_list_P, J_list_P, q_P_list


def calculate_efficiency_N(Tc, Th, n, l):
    eff_list_N = []
    J_list_N = []
    dx = l / (n - 1)
    q_N_list = []
    for j in range(0, 51, 1):
        J = j
        T = temperature_distribution_N(n, J, Tc, Th, 10)
        sb, res, th = material_N(T)
        c1 = J * sb / th
        c2 = -1 / th
        c3 = sb ** 2 * J ** 2 / th
        c4 = -J * sb / th
        c5 = res * J ** 2

        # 计算热流密度, 定义为 q_N表示N的热流密度
        q_N = np.zeros(n)
        for k in range(1, n):
            q_N[k] = ((1 / dx - c1[k]) * T[k] - T[k - 1] / dx) / (c2[k])
        q_N[0] = (1 - c4[1] * dx) * q_N[1] - c3[1] * dx * T[1] - c5[1] * dx
        # 将每次的q_N(L)给存起来
        q_N_list.append(q_N[-1])
        # 第一个积分
        Cumulative_scoring1 = 0
        # 第二个积分
        Cumulative_scoring2 = 0
        for m in range(1, n):
            T1 = T[m]
            T2 = T[m - 1]
            Cumulative_scoring1 += (sb[m] + sb[m - 1]) / 2 * (T1 - T2)
            Cumulative_scoring2 += (res[m] + res[m - 1]) / 2 * dx

        eff = J * (Cumulative_scoring1 + J * Cumulative_scoring2) / q_N[n - 1]
        eff_list_N.append(eff)
        J_list_N.append(J)
    return eff_list_N, J_list_N, q_N_list

eff_list_P, J_list_P, q_P_list = calculate_efficiency_P(Tc, Th, 10, 1)
eff_list_N, J_list_N, q_N_list = calculate_efficiency_N(Tc, Th, 10, 1)


def calculate_total(area_ratio):
    eff_total_list = []
    I_list = []
    for m in range(1, 21, 1):  # m代表电流密度，在P型中由于单位横截面积就是1cm^2，故用电流密度代替电流
        I_list.append(m)
        index_N = int(m * area_ratio)
        if m < len(eff_list_P) and index_N < len(eff_list_N):
            denominator = q_P_list[m] / m - q_N_list[m] / (m / area_ratio)
            if denominator != 0:
                eff_total = (eff_list_P[m - 1] * q_P_list[m] / m - eff_list_N[index_N] * q_N_list[m] / (m / area_ratio)) / denominator
                eff_total_list.append(eff_total)
    return I_list, eff_total_list


I_list, eff_total_list = calculate_total(area_ratio)

def Power_total(area_ratio):
    Power_total_list = []
    I_list, eff_total_list = calculate_total(area_ratio)
    for n in range(0, len(eff_total_list)):
        pow_total = eff_total_list[n] * (-q_P_list[n+1] - q_N_list[n+1] * area_ratio)
        Power_total_list.append(pow_total)
    return Power_total_list

def plot_data(x_list, y_list, x_label, y_label, title, filename=None):
    plt.figure(figsize=(8, 6))
    plt.plot(x_list, y_list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename)
        print(f"图表已保存为 {filename}")
    else:
        plt.show()

# 主程序入口
if __name__ == "__main__":
    # 计算效率和功率
    eff_list_P, J_list_P, q_P_list = calculate_efficiency_P(Tc, Th, 10, 1)
    eff_list_N, J_list_N, q_N_list = calculate_efficiency_N(Tc, Th, 10, 1)
    
    print(f"计算总效率...")
    I_list, eff_total_list = calculate_total(area_ratio)
    
    print(f"计算总功率...")
    Power_total_list = Power_total(area_ratio)
    
    # 绘制并保存P型效率曲线
    p_output_file = f"{output_dir}/P_{value_p}_efficiency.png"
    plot_data(J_list_P, eff_list_P, 'J (A/m2)', 'eff', f'P型材料(组分={value_p})效率曲线', p_output_file)
    
    # 绘制并保存N型效率曲线
    n_output_file = f"{output_dir}/N_{value_n}_efficiency.png"
    plot_data(J_list_N, eff_list_N, 'J (A/m2)', 'eff', f'N型材料(组分={value_n})效率曲线', n_output_file)
    
    # 绘制并保存总效率曲线
    total_eff_output_file = f"{output_dir}/total_efficiency_{value_p}_{value_n}.png"
    plot_data(I_list, eff_total_list, 'I (A)', 'eff', f'总效率曲线 (P:{value_p}, N:{value_n}, 面积比:{area_ratio})', total_eff_output_file)
    
    # 绘制并保存总功率曲线
    power_output_file = f"{output_dir}/total_power_{value_p}_{value_n}.png"
    plot_data(I_list, Power_total_list, 'I (A)', 'Power (W/m2)', f'总功率曲线 (P:{value_p}, N:{value_n}, 面积比:{area_ratio})', power_output_file)
    
    # 绘制并保存效率-功率关系图
    eff_power_output_file = f"{output_dir}/efficiency_power_{value_p}_{value_n}.png"
    plot_data(Power_total_list, eff_total_list, 'Power (W/m2)', 'eff', f'效率-功率关系 (P:{value_p}, N:{value_n}, 面积比:{area_ratio})', eff_power_output_file)
    
    # 保存计算结果到CSV文件
    try:
        import pandas as pd
        
        # 创建P型材料数据DataFrame
        p_data = pd.DataFrame({
            'J_P': J_list_P,
            'eff_P': eff_list_P,
            'q_P': q_P_list
        })
        p_data.to_csv(f"{output_dir}/P_{value_p}_data.csv", index=False)
        
        # 创建N型材料数据DataFrame
        n_data = pd.DataFrame({
            'J_N': J_list_N,
            'eff_N': eff_list_N,
            'q_N': q_N_list
        })
        n_data.to_csv(f"{output_dir}/N_{value_n}_data.csv", index=False)
        
        # 创建总效率和功率数据DataFrame
        total_data = pd.DataFrame({
            'I': I_list,
            'eff_total': eff_total_list,
            'Power_total': Power_total_list
        })
        total_data.to_csv(f"{output_dir}/total_data_{value_p}_{value_n}.csv", index=False)
        
        print(f"数据已保存到CSV文件")
    except Exception as e:
        print(f"保存数据到CSV文件时出错: {str(e)}")
    
    # 打印最大效率和功率
    if eff_total_list:
        max_eff = max(eff_total_list)
        max_eff_idx = eff_total_list.index(max_eff)
        print(f"\n最大效率: {max_eff:.6f} (电流: {I_list[max_eff_idx]}A)")
    
    if Power_total_list:
        max_power = max(Power_total_list)
        max_power_idx = Power_total_list.index(max_power)
        print(f"最大功率: {max_power:.6e} W/m2 (电流: {I_list[max_power_idx]}A)")
    
    # 创建综合结果图
    try:
        plt.figure(figsize=(12, 10))
        
        # 创建2x2子图
        plt.subplot(2, 2, 1)
        plt.plot(J_list_P, eff_list_P, 'b-o')
        plt.xlabel('电流密度 (A/m2)')
        plt.ylabel('效率')
        plt.title(f'P型材料(组分={value_p})效率')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(J_list_N, eff_list_N, 'r-o')
        plt.xlabel('电流密度 (A/m2)')
        plt.ylabel('效率')
        plt.title(f'N型材料(组分={value_n})效率')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(I_list, eff_total_list, 'g-o')
        plt.xlabel('电流 (A)')
        plt.ylabel('效率')
        plt.title(f'总效率 (面积比={area_ratio})')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(I_list, Power_total_list, 'm-o')
        plt.xlabel('电流 (A)')
        plt.ylabel('功率 (W/m2)')
        plt.title('总功率')
        plt.grid(True)
        
        plt.tight_layout()
        summary_output_file = f"{output_dir}/summary_{value_p}_{value_n}.png"
        plt.savefig(summary_output_file)
        print(f"综合结果图已保存为 {summary_output_file}")
        
    except Exception as e:
        print(f"创建综合结果图时出错: {str(e)}")
    
    print("\n计算完成!")
    #绘制总效率随着电流的改变
plot_data(I_list, eff_total_list,'I','eff','total')

#绘制总功率随电流的改变而改变
Power_total_list = Power_total(area_ratio)
plot_data(I_list, Power_total_list,'I','Power','total')

#绘制效率随功率的关系图
plot_data( Power_total_list, eff_total_list,'Power','eff','total')