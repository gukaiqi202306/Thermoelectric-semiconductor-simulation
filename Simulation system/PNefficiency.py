import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='半导体热电材料效率计算程序')
    parser.add_argument('--p_composition', type=str, required=True, help='P型材料组分')
    parser.add_argument('--n_composition', type=str, required=True, help='N型材料组分')
    parser.add_argument('--Tc', type=float, default=300, help='冷端温度 (K)')
    parser.add_argument('--Th', type=float, default=500, help='热端温度 (K)')
    parser.add_argument('--n_points', type=int, default=10, help='网格点数')
    parser.add_argument('--output_dir', type=str, default="efficiency_results", help='输出目录')
    return parser.parse_args()

# 根据材料类型和组分选择数据文件
def get_data_filename(material_type, composition):
    if material_type == 'P':
        if composition == '0.01':
            return 'P_yuanshi_2_5.xls'
        elif composition == '0.02':
            return 'P_yuanshi_3_1.xls'
        elif composition == '0.03':
            return 'P_yuanshi_3_7.xls'
        else:
            print(f"错误: 不支持的P型材料组分 {composition}")
            sys.exit(1)
    elif material_type == 'N':
        if composition == '0.0004':
            return 'N_yuanshi_0.0004.xls'
        elif composition == '0.0012':
            return 'N_yuanshi_0.0012.xls'
        elif composition == '0.0020':
            return 'N_yuanshi_0.0020.xls'
        else:
            print(f"错误: 不支持的N型材料组分 {composition}")
            sys.exit(1)
    else:
        print(f"错误: 不支持的材料类型 {material_type}")
        sys.exit(1)

# 读取材料数据
def load_material_data(material_type, composition):
    filename = get_data_filename(material_type, composition)
    try:
        data = pd.read_excel(filename, header=None).values
        print(f"成功加载{material_type}型材料(组分={composition})数据")
        print("温度\t塞贝克系数\t电导率\tZT")
        for row in data[:5]:
            print(f"{row[0]:.1f}\t{row[1]:.2e}\t{row[2]:.2e}\t{row[4]:.2f}")
        return data
    except Exception as e:
        print(f"加载材料数据错误: {str(e)}")
        sys.exit(1)

# 材料属性计算函数 - 根据材料类型动态选择
def material_properties(T, material_data, material_type):
    if material_type == 'P':
        sb = CubicSpline(material_data[:, 0], material_data[:, 1])(T) * 1e-6  # 塞贝克系数
        res = CubicSpline(material_data[:, 2], material_data[:, 3])(T) * 1e-3  # 电阻率
        ZT = CubicSpline(material_data[:, 4], material_data[:, 5])(T)  # 优值系数
        th = (T * sb ** 2) / (ZT * res)  # 热导率
        return sb, res, th
    elif material_type == 'N':
        sb = CubicSpline(material_data[:, 0], material_data[:, 1])(T) * 1e-6  # 塞贝克系数，单位：V/K
        res = CubicSpline(material_data[:, 2], material_data[:, 3])(T) * 1e-3  # 电导率，单位：S/m
        th = CubicSpline(material_data[:, 4], material_data[:, 5])(T)/100  # 热导率，单位：W/(m·K)
        return sb, res, th

# 温度分布计算函数
def temperature_distribution(n, J, Tc, Th, max_iter, material_data, material_type):
    l = 1
    dx = l / (n - 1)
    T = np.linspace(Tc, Th, n)
    
    for _ in range(max_iter):
        A = np.zeros((n, n))
        b = np.zeros(n)
        sb, res, th = material_properties(T, material_data, material_type)
        
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
        for i in range(1, n-1):
            A[i, i-1] = 1/(c2[i] * dx)
            A[i, i] = c4[i+1]/c2[i+1] - 1/(c2[i+1] * dx) - (1 - c1[i] * dx)/(c2[i] * dx)
            A[i, i+1] = (1 - c1[i+1] * dx)/(c2[i+1] * dx) - c3[i+1] * dx - (1 - c1[i+1] * dx) * c4[i+1]/c2[i+1]
            b[i] = c5[i-1] * dx
            
        try:
            T_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("线性方程组求解失败")
            return None
            
        T = T_new.copy()
        
    return T

# 效率计算函数
def calculate_efficiency(Tc, Th, n, l, material_data, material_type):
    """计算材料效率"""
    eff_list = []
    J_list = []
    dx = l / (n - 1)
    
    if material_type == 'P':
        # P型材料电流密度范围为-30到0，步长2
        current_range = range(0, 31, 2)
        sign = -1  # P型为负电流
    else:  # N型
        # N型材料电流密度范围为0到50，步长1
        current_range = range(0, 51, 1)
        sign = 1  # N型为正电流
    
    for j in current_range:
        J = sign * j
        print(f"\n计算{material_type}型效率 (J={J}A/m^2)")
        
        T = temperature_distribution(n, J, Tc, Th, 10, material_data, material_type)
        if T is None:
            print("温度分布计算失败")
            continue
            
        sb, res, th = material_properties(T, material_data, material_type)
        
        # 计算热流系数
        c1 = J * sb / th
        c2 = -1 / th
        c3 = sb ** 2 * J ** 2 / th
        c4 = -J * sb / th
        c5 = res * J ** 2
        
        # 计算热流密度q
        q = np.zeros(n)
        for k in range(1, n):
            q[k] = ((1/dx - c1[k]) * T[k] - T[k-1]/dx) / (c2[k])
        q[0] = (1 - c4[1] * dx) * q[1] - c3[1] * dx * T[1] - c5[1] * dx
        
        # 计算积分项
        seebeck_integral = 0
        resistivity_integral = 0
        for m in range(1, n):
            T1 = T[m]
            T2 = T[m-1]
            seebeck_integral += (sb[m] + sb[m-1]) / 2 * (T1 - T2)
            resistivity_integral += (res[m] + res[m-1]) / 2 * dx
            
        print(f"塞贝克积分: {seebeck_integral:.6f} V")
        print(f"电阻率积分: {resistivity_integral:.6f} Ω·m")
        
        # 计算效率
        if q[n-1] != 0:
            eff = J * (seebeck_integral + J * resistivity_integral) / q[n-1]
            print(f"计算效率: {eff:.6f}")
            
            # 检查效率是否超过卡诺效率
            carnot_eff = (Th - Tc) / Th
            if abs(eff) > carnot_eff:
                print(f"警告: 计算效率 {eff:.6f} 超过卡诺效率 {carnot_eff:.6f}")
                
            eff_list.append(eff)
            J_list.append(J)
        else:
            print("热流为零，跳过此点")
            
    return eff_list, J_list

# 主程序
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 设置温度范围
    Tc = args.Tc
    Th = args.Th
    n = args.n_points
    l = 1   # 材料长度
    
    # 获取材料组分
    p_composition = args.p_composition
    n_composition = args.n_composition
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"计算P型材料(组分={p_composition})的效率曲线")
    print(f"温度范围: Tc={Tc}K, Th={Th}K")
    
    # 加载材料数据
    p_material_data = load_material_data('P', p_composition)
    
    # 计算效率
    p_eff_list, p_J_list = calculate_efficiency(Tc, Th, n, l, p_material_data, 'P')
    
    print(f"\nP型效率数据:", p_eff_list)
    print(f"P型电流密度:", p_J_list)
    
    print(f"\n计算N型材料(组分={n_composition})的效率曲线")
    print(f"温度范围: Tc={Tc}K, Th={Th}K")
    
    # 加载材料数据
    n_material_data = load_material_data('N', n_composition)
    
    # 计算效率
    n_eff_list, n_J_list = calculate_efficiency(Tc, Th, n, l, n_material_data, 'N')
    
    # 输出结果
    print(f"\nN型效率数据:", n_eff_list)
    print(f"N型电流密度:", n_J_list)
    
    # 绘制单独的P型效率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(p_J_list, p_eff_list, 'b-o')
    plt.xlabel('电流密度 (A/m^2)')
    plt.ylabel('效率')
    plt.title(f'P型材料(组分={p_composition})效率曲线')
    plt.grid(True)
    
    # 保存P型效率曲线图像
    p_output_file = f"{output_dir}/P_{p_composition}_efficiency.png"
    plt.savefig(p_output_file)
    print(f"\nP型效率曲线已保存为 {p_output_file}")
    
    # 绘制单独的N型效率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(n_J_list, n_eff_list, 'r-o')
    plt.xlabel('电流密度 (A/m^2)')
    plt.ylabel('效率')
    plt.title(f'N型材料(组分={n_composition})效率曲线')
    plt.grid(True)
    
    # 保存N型效率曲线图像
    n_output_file = f"{output_dir}/N_{n_composition}_efficiency.png"
    plt.savefig(n_output_file)
    print(f"\nN型效率曲线已保存为 {n_output_file}")
    
    # 显示图形
    plt.show() 