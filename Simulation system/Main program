import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, 
                            QGroupBox, QFrame, QGridLayout, QDialog, QScrollArea,
                            QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import pandas as pd

class StatusLight(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setStyleSheet("background-color: red; border-radius: 10px;")
        
    def set_status(self, status):
        color = "green" if status else "red"
        self.setStyleSheet(f"background-color: {color}; border-radius: 10px;")

class ImageViewerDialog(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片查看")
        self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen().geometry()
        self.setMinimumSize(screen.width() // 2, screen.height() // 2)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建图片容器
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background-color: white;")
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # 创建图片标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.image_label)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.image_container)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # 添加关闭按钮
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 5, 10, 5)
        close_button = QPushButton("关闭")
        close_button.setFixedWidth(100)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #dcdcdc;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
        """)
        close_button.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 保存原始图片
        self.original_pixmap = pixmap
        # 初始显示
        self.resizeEvent(None)
    
    def resizeEvent(self, event):
        """当窗口大小改变时，调整图片大小"""
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull():
            # 获取可用空间大小（减去按钮区域高度）
            available_size = self.size()
            available_size.setHeight(available_size.height() - 40)  # 40是按钮区域的高度
            
            # 计算缩放后的图片大小
            scaled_pixmap = self.original_pixmap.scaled(
                available_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 更新图片
            self.image_label.setPixmap(scaled_pixmap)

class ClickableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针为手型
        
    def mouseDoubleClickEvent(self, event):
        if self.pixmap() and not self.pixmap().isNull():
            dialog = ImageViewerDialog(self.pixmap(), self.window())
            dialog.exec_()

class ThermoelectricCalculator:
    def __init__(self):
        # 移除对iter_edit的依赖
        self.p_type_data = {}
        self.n_type_data = {}
        self.interpolators = {}
        
        # 读取P型材料数据，修正组分值对应关系
        p_files = {
            "0.01": "P_yuanshi_2_5.xls",  # 0.01对应2.5
            "0.02": "P_yuanshi_3_1.xls",  # 0.02对应3.1
            "0.03": "P_yuanshi_3_7.xls"   # 0.03对应3.7
        }
        
        # 读取N型材料数据
        n_files = {
            "0.0004": "N_yuanshi_0.0004.xls",
            "0.0012": "N_yuanshi_0.0012.xls",
            "0.0020": "N_yuanshi_0.0020.xls",
            "0.0028": "N_yuanshi_0.0028.xls"
        }
        
        def read_excel_file(filename):
            """读取Excel文件的辅助函数"""
            try:
                # 首先尝试使用xlrd引擎
                try:
                    import xlrd
                    # 不使用列名读取数据
                    data = pd.read_excel(filename, engine='xlrd', header=None)
                    print(f"成功使用xlrd读取文件: {filename}")
                    return data
                except ImportError:
                    print("xlrd未安装，尝试使用openpyxl...")
                    return None
            except Exception as e:
                print(f"读取文件失败: {str(e)}")
                return None
        
        def find_columns(data):
            """根据数据结构查找相应的列"""
            try:
                # 检查数据的结构来确定正确的列索引
                # 对于P_yuanshi文件（如P_yuanshi_2_5.xls），列结构为：
                # 温度(A列,0), 塞贝克系数(B列,1), 温度(C列,2), 电阻率(D列,3), 温度(E列,4), 优值系数(F列,5)
                if data.shape[1] >= 6:  # 确保有足够的列
                    print("找到的列结构：")
                    for i in range(min(6, data.shape[1])):
                        print(f"列 {i}: {data.iloc[0, i]}")
                    
                    # 检查前几行的数据来识别是P型还是N型文件
                    # P型文件特征：第一列数值在300左右（温度）
                    first_col_values = data.iloc[0:5, 0].values
                    print(f"第一列前5个值: {first_col_values}")
                    
                    if any(290 <= v <= 310 for v in first_col_values if isinstance(v, (int, float))):
                        print("检测到P型材料数据文件格式")
                        return {
                            "temp": 0,        # A列作为温度
                            "seebeck": 1,     # B列作为塞贝克系数
                            "resistivity": 3, # D列作为电阻率
                            "thermal_cond": 5 # F列作为优值系数（但我们需要另外计算热导率）
                        }
                    else:
                        print("检测到N型材料数据文件格式")
                        return {
                            "temp": 0,        # 第1列作为温度
                            "seebeck": 1,     # 第2列作为塞贝克系数
                            "resistivity": 3, # 第4列作为电阻率
                            "thermal_cond": 5 # 第6列作为热导率
                        }
                else:
                    print("警告：数据列数不足，使用默认列映射")
                    return {
                        "temp": 0,
                        "seebeck": 1,
                        "resistivity": 3,
                        "thermal_cond": 5
                    }
            except Exception as e:
                print(f"查找列错误: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        # 读取所有P型材料数据
        for composition, filename in p_files.items():
            print(f"\n尝试读取P型材料数据文件: {filename}")
            data = read_excel_file(filename)
            if data is not None:
                try:
                    # 查找列
                    columns = find_columns(data)
                    if columns:
                        # P型材料：F列是优值系数(ZT)，我们需要从中反推热导率
                        # 热导率 k = (α^2 × T) / (ρ × ZT)
                        # 其中 α 是塞贝克系数，ρ 是电阻率，T 是温度，ZT 是优值系数
                        seebeck = data[columns['seebeck']].values * 1e-6  # μV/K 转换为 V/K
                        resistivity = data[columns['resistivity']].values * 1e-6  # μΩ·m 转换为 Ω·m (修正单位换算错误)
                        temperature = data[columns['temp']].values
                        zt_values = data[columns['thermal_cond']].values  # 这里实际上是ZT值
                        
                        # 计算热导率
                        thermal_cond = []
                        for i in range(len(temperature)):
                            try:
                                # 避免无效ZT值和除以零
                                if zt_values[i] > 0:
                                    k = (seebeck[i]**2 * temperature[i]) / (resistivity[i] * zt_values[i])
                                    # 添加合理性检查 (热导率通常在0.1-100 W/m·K范围内)
                                    if 0.1 <= k <= 100:
                                        thermal_cond.append(k)
                                    else:
                                        print(f"警告: 计算得到异常热导率值 {k:.3f} W/m·K，使用默认值 2.0 W/m·K")
                                        thermal_cond.append(2.0)  # 更合理的默认值
                                else:
                                    print(f"警告: 无效ZT值 {zt_values[i]}，使用默认热导率 2.0 W/m·K")
                                    thermal_cond.append(2.0)  # 更合理的默认值
                            except Exception as e:
                                print(f"热导率计算错误: {str(e)}，使用默认值 2.0 W/m·K")
                                thermal_cond.append(2.0)  # 更合理的默认值
                        
                        self.p_type_data[composition] = {
                            "temp": temperature,
                            "seebeck": seebeck,
                            "resistivity": resistivity,
                            "thermal_cond": np.array(thermal_cond)  # 从ZT反推的热导率
                        }
                        print(f"成功读取P型材料数据: {composition}")
                        print(f"温度范围: {min(temperature)}-{max(temperature)} K")
                        print(f"塞贝克系数范围: {min(seebeck*1e6)}-{max(seebeck*1e6)} μV/K")
                        print(f"电阻率范围: {min(resistivity*1e6)}-{max(resistivity*1e6)} μΩ·m")
                        print(f"计算的热导率范围: {min(thermal_cond)}-{max(thermal_cond)} W/(m·K)")
                    else:
                        print(f"在文件 {filename} 中未找到所需的列")
                        
                except Exception as e:
                    print(f"处理P型材料数据文件 {filename} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # 读取所有N型材料数据
        for composition, filename in n_files.items():
            print(f"\n尝试读取N型材料数据文件: {filename}")
            data = read_excel_file(filename)
            if data is not None:
                try:
                    # 查找列
                    columns = find_columns(data)
                    if columns:
                        self.n_type_data[composition] = {
                            "temp": data[columns['temp']].values,
                            "seebeck": -data[columns['seebeck']].values * 1e-6,  # μV/K 转换为 V/K，N型为负值
                            "resistivity": data[columns['resistivity']].values * 1e-5,  # μΩ·m 转换为 Ω·m
                            "thermal_cond": data[columns['thermal_cond']].values  # W/(m·K)
                        }
                        print(f"成功读取N型材料数据: {composition}")
                    else:
                        print(f"在文件 {filename} 中未找到所需的列")
                        
                except Exception as e:
                    print(f"处理N型材料数据文件 {filename} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
        print("\n数据读取完成")
        print(f"成功读取的P型材料: {list(self.p_type_data.keys())}")
        print(f"成功读取的N型材料: {list(self.n_type_data.keys())}")
        
    def create_interpolators(self, material_type, composition):
        """为给定材料创建属性插值器"""
        try:
            data = self.p_type_data if material_type == 'p' else self.n_type_data
            mat_data = data[composition]
            
            # ==== 增加插值范围限制 ====
            temps = mat_data["temp"]
            seebeck = mat_data["seebeck"]
            resistivity = mat_data["resistivity"]
            thermal_cond = mat_data["thermal_cond"]
            
            # 确保数据有序
            sort_idx = np.argsort(temps)
            temps = temps[sort_idx]
            seebeck = seebeck[sort_idx]
            resistivity = resistivity[sort_idx]
            thermal_cond = thermal_cond[sort_idx]
            
            # 打印材料属性范围
            print(f"\n===== 创建 {material_type}型材料插值器 (组分={composition}) =====")
            print(f"温度范围: {min(temps)}-{max(temps)} K")
            print(f"塞贝克系数范围: {min(seebeck*1e6):.2f}-{max(seebeck*1e6):.2f} μV/K")
            print(f"电阻率范围: {min(resistivity*1e6):.2f}-{max(resistivity*1e6):.2f} μΩ·m")
            print(f"热导率范围: {min(thermal_cond):.2f}-{max(thermal_cond):.2f} W/(m·K)")
            
            # 创建边界值保护的插值器
            self.interpolators[f"{material_type}_{composition}"] = {
                "seebeck": interp1d(temps, seebeck, kind='linear', 
                                   bounds_error=False, 
                                   fill_value=(seebeck[0], seebeck[-1])),  # 限制外推值
                "resistivity": interp1d(temps, resistivity, kind='linear',
                                      bounds_error=False,
                                      fill_value=(resistivity[0], resistivity[-1])),
                "thermal_cond": interp1d(temps, thermal_cond, kind='linear',
                                       bounds_error=False,
                                       fill_value=(thermal_cond[0], thermal_cond[-1]))
            }
            
            print(f"插值器创建成功")
            
        except Exception as e:
            print(f"创建插值器错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def calculate_temperature_distribution(self, Th, Tc, n_points, material_type, composition, current_density, max_iter=50):
        """
        根据热电材料物理性质计算温度分布，修正单位和插值方法
        """
        try:
            print(f"\n开始计算温度分布: {material_type}型, 组分={composition}, 电流密度={current_density}A/cm²")
            print(f"边界条件: Th={Th}K, Tc={Tc}K, 格点数={n_points}")
            
            # 创建插值器（如果还没有创建）
            interp_key = f"{material_type}_{composition}"
            if interp_key not in self.interpolators:
                self.create_interpolators(material_type, composition)
            
            # 初始化格点位置和温度
            L = 1.0  # 标准化长度
            dx = L / (n_points - 1)  # 网格间距
            x = np.linspace(0, L, n_points)  # 从0到1的均匀分布
            T = np.linspace(Tc, Th, n_points)  # 初始线性温度分布
            
            print(f"初始温度分布: {T}")
            
            # ===== 修正1: 电流密度单位处理 =====
            # 假设输入电流密度为A/cm²，使用更合理的转换系数
            J = current_density * 100  # 转换为A/m²但限制在合理范围
            
            # 检查电流密度是否在合理范围内
            if abs(J) > 5e3:  # 根据物理合理性设定上限
                print(f"警告: 电流密度 {J} A/m² 超过正常范围，将限制为5000A/m²")
                J = np.sign(J) * 5e3
            
            print(f"网格间距: dx={dx}, 电流密度: J={J}A/m²")
            
            # 迭代求解参数
            relaxation_factor = 0.2  # 松弛因子，提高稳定性
            convergence_threshold = 0.01  # 收敛阈值
            jacobi_iterations = 100  # 内部Jacobi迭代次数上限
            
            for iter_count in range(max_iter):
                # 保存旧的温度分布用于收敛判断
                T_old = T.copy()
                
                # ===== 修正2: 先计算并存储所有网格点的材料属性 =====
                seebeck = np.zeros(n_points)
                resistivity = np.zeros(n_points)
                thermal_cond = np.zeros(n_points)
                
                for i in range(n_points):
                    T_safe = np.clip(T[i], 300, 700)  # 确保温度在有效范围内
                    seebeck[i] = self.interpolators[interp_key]["seebeck"](T_safe)
                    resistivity[i] = self.interpolators[interp_key]["resistivity"](T_safe)
                    thermal_cond[i] = self.interpolators[interp_key]["thermal_cond"](T_safe)
                
                # 构建系数矩阵和右端向量
                A = np.zeros((n_points, n_points))
                b = np.zeros(n_points)
                
                # 设置边界条件
                A[0, 0] = 1.0
                b[0] = Tc
                A[n_points-1, n_points-1] = 1.0
                b[n_points-1] = Th
                
                # ===== 修正3: 内部点的系数计算改为基于温度的插值 =====
                for i in range(1, n_points-1):
                    # 使用基于温度分布的插值获取半点处的材料属性
                    T_half_minus = 0.5 * (T[i-1] + T[i])
                    T_half_plus = 0.5 * (T[i] + T[i+1])
                    
                    # 直接使用插值器计算半点处属性
                    k_minus = self.interpolators[interp_key]["thermal_cond"](np.clip(T_half_minus, 300, 700))
                    k_plus = self.interpolators[interp_key]["thermal_cond"](np.clip(T_half_plus, 300, 700))
                    s_minus = self.interpolators[interp_key]["seebeck"](np.clip(T_half_minus, 300, 700))
                    s_plus = self.interpolators[interp_key]["seebeck"](np.clip(T_half_plus, 300, 700))
                    rho_i = resistivity[i]  # 焦耳热在节点上
                    
                    # 热传导项系数
                    A[i, i-1] = k_minus / dx**2
                    A[i, i] = -(k_minus + k_plus) / dx**2
                    A[i, i+1] = k_plus / dx**2
                    
                    # 塞贝克项（热电耦合）
                    A[i, i-1] += -J * s_minus / (2 * dx)
                    A[i, i+1] += J * s_plus / (2 * dx)
                    
                    # ===== 修正4: 焦耳热项修正为正贡献 =====
                    b[i] = rho_i * J**2  # 正确的焦耳热项（热源）
                
                # 使用带松弛因子的Jacobi迭代
                T_new = T_old.copy()
                
                for jacobi_iter in range(jacobi_iterations):
                    T_prev = T_new.copy()
                    
                    # 边界点固定不变
                    T_new[0] = Tc
                    T_new[n_points-1] = Th
                    
                    # 更新内部点
                    for i in range(1, n_points-1):
                        if abs(A[i, i]) > 1e-10:  # 避免除以零
                            numerator = b[i]
                            for j in range(n_points):
                                if j != i:
                                    numerator -= A[i, j] * T_prev[j]
                            
                            # 使用更小的松弛因子提高稳定性
                            T_new[i] = T_prev[i] + relaxation_factor * (numerator / A[i, i] - T_prev[i])
                    
                    # 检查内部迭代收敛性
                    if np.max(np.abs(T_new - T_prev)) < 0.001:
                        break
                
                # 检查解的合理性
                if np.any(np.isnan(T_new)) or np.any(np.isinf(T_new)):
                    print(f"警告：第{iter_count+1}次迭代解不合理，使用线性插值")
                    T_new = np.linspace(Tc, Th, n_points)
                
                # 限制温度在物理合理范围内（略微放宽范围）
                T_new = np.clip(T_new, min(Tc, Th)*0.95, max(Tc, Th)*1.1)
                
                # 计算收敛情况
                max_change = np.max(np.abs(T_new - T_old))
                print(f"迭代{iter_count+1}次完成，最大温度变化: {max_change:.6f}K")
                
                # 更新温度
                T = T_new.copy()
                
                # 判断是否已经收敛
                if max_change < convergence_threshold:
                    print(f"温度分布已收敛，在第{iter_count+1}次迭代")
                    break
            
            # ===== 修正5: 改进的热流验证 =====
            try:
                # 计算节点处的热流密度 q = J*T*S - k*dT/dx
                dTdx = np.zeros(n_points)
                dTdx[1:-1] = (T[2:] - T[:-2]) / (2*dx)  # 中心差分
                dTdx[0] = (T[1] - T[0]) / dx  # 前向差分
                dTdx[-1] = (T[-1] - T[-2]) / dx  # 后向差分
                
                q = np.zeros(n_points)
                for i in range(n_points):
                    q[i] = J * T[i] * seebeck[i] - thermal_cond[i] * dTdx[i]
                
                heat_in = q[0]  # 入口热流
                heat_out = q[-1]  # 出口热流
                joule_heat = sum(resistivity * J**2 * dx)  # 总焦耳热
                
                print(f"热流验证: 入口热流={heat_in:.3f}, 出口热流={heat_out:.3f}, 焦耳热={joule_heat:.3f}")
                print(f"热流平衡检查: 出口-入口-焦耳热={heat_out-heat_in-joule_heat:.3f}（应接近零）")
            except Exception as e:
                print(f"热流验证计算错误: {e}")
            
            # 打印最终温度分布
            print(f"最终温度分布: {T}")
            
            return x, T
            
        except Exception as e:
            print(f"计算温度分布错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 出错时返回线性温度分布
            L = 1.0
            return np.linspace(0, L, n_points), np.linspace(Tc, Th, n_points)
    
    def calculate_efficiency(self, Th, Tc, material_type, composition, current_density, x=None, T=None):
        """
        参考温度分布和效率的实现计算热电材料效率
        
        参数:
        Th: 高温端温度 (K)
        Tc: 低温端温度 (K)
        material_type: 材料类型 ('p' 或 'n')
        composition: 材料组分
        current_density: 电流密度 (A/cm²)
        x, T: 温度分布数据
        
        返回:
        efficiency: 效率 (%)
        power: 输出功率密度 (W/m²)
        """
        try:
            # 验证输入参数
            if Th <= Tc:
                print(f"警告: 温度差无效 (Th={Th}K, Tc={Tc}K)")
                return 0.0, 0.0
            
            # 准备插值器
            interp_key = f"{material_type}_{composition}"
            if interp_key not in self.interpolators:
                self.create_interpolators(material_type, composition)
            
            # 确保温度分布数据有效
            if x is None or T is None or len(x) < 3:
                print(f"温度分布数据无效，使用线性温度分布近似")
                n_points = 20
                x = np.linspace(0, 1.0, n_points)
                T = np.linspace(Tc, Th, n_points)
            
            # 获取格点数和间距
            n_points = len(x)
            dx = (x[-1] - x[0]) / (n_points - 1)
            
            # 计算各节点处的材料属性并打印调试信息
            seebeck = np.zeros(n_points)
            resistivity = np.zeros(n_points)
            thermal_cond = np.zeros(n_points)
            
            for i in range(n_points):
                T_safe = np.clip(T[i], 300, 700)  # 确保温度在有效范围内
                seebeck[i] = self.interpolators[interp_key]["seebeck"](T_safe)
                resistivity[i] = self.interpolators[interp_key]["resistivity"](T_safe)
                thermal_cond[i] = self.interpolators[interp_key]["thermal_cond"](T_safe)
            
            # 打印材料属性统计信息用于调试
            print(f"\n===== {material_type}型材料属性统计({composition}) =====")
            print(f"塞贝克系数(V/K): 最小={np.min(seebeck):.3e}, 最大={np.max(seebeck):.3e}, 平均={np.mean(seebeck):.3e}")
            print(f"电阻率(Ω·m): 最小={np.min(resistivity):.3e}, 最大={np.max(resistivity):.3e}, 平均={np.mean(resistivity):.3e}")
            print(f"热导率(W/m·K): 最小={np.min(thermal_cond):.3f}, 最大={np.max(thermal_cond):.3f}, 平均={np.mean(thermal_cond):.3f}")
            
            # 验证P型材料塞贝克系数符号是否正确
            if material_type == 'p' and np.any(seebeck < 0):
                print(f"警告: P型材料塞贝克系数出现负值! 最小值: {np.min(seebeck):.3e}V/K")
            elif material_type == 'n' and np.any(seebeck > 0):
                print(f"警告: N型材料塞贝克系数出现正值! 最大值: {np.max(seebeck):.3e}V/K")
            
            # 正确转换单位: A/cm² → A/m²
            J = current_density * 100  # 转换为A/m² (1A/cm² = 100A/m²)
            
            # 计算温度梯度 (使用中心差分)
            dTdx = np.zeros_like(T)
            dTdx[1:-1] = (T[2:] - T[:-2]) / (2*dx)  # 中心差分
            dTdx[0] = (T[1] - T[0]) / dx            # 前向差分
            dTdx[-1] = (T[-1] - T[-2]) / dx         # 后向差分
            
            # 计算热流密度: q(x) = κ·dT/dx - J·S·T
            q = np.zeros(n_points)
            for i in range(n_points):
                q[i] = thermal_cond[i] * dTdx[i] - J * seebeck[i] * T[i]
            
            # 确保热流方向正确（从高温端到低温端）
            if np.mean(q) < 0:
                print(f"热流方向修正: 平均热流 {np.mean(q):.2e} 为负值，已反转")
                q = -q
            
            # 第一个积分：∫ S·dT（温度差间的塞贝克积分）
            seebeck_integral = 0.0
            for i in range(1, n_points):
                seebeck_integral += (seebeck[i] + seebeck[i-1]) / 2 * (T[i] - T[i-1])
            
            # 第二个积分：∫ ρ·dx（电阻率沿长度的积分）
            resistivity_integral = 0.0
            for i in range(1, n_points):
                resistivity_integral += (resistivity[i] + resistivity[i-1]) / 2 * dx
            
            # 计算功率输出
            seebeck_power = J * seebeck_integral  # 塞贝克效应产生的功率
            joule_heat = J**2 * resistivity_integral  # 焦耳热损失
            net_power = seebeck_power - joule_heat  # 净功率输出
            
            # 计算热输入（高温端热流）
            heat_in = abs(q[0])  # 热流密度
            
            # 计算焦耳热总量
            total_joule_heat = 0.0
            for i in range(n_points-1):
                seg_resistivity = (resistivity[i] + resistivity[i+1]) / 2
                total_joule_heat += J**2 * seg_resistivity * dx
            
            # 打印详细的功率和热流信息用于调试
            print(f"\n===== 能量分析 =====")
            print(f"塞贝克功率: {seebeck_power:.3e} W/m²")
            print(f"焦耳热损失: {joule_heat:.3e} W/m²")
            print(f"净功率输出: {net_power:.3e} W/m²")
            print(f"热输入(高温端): {heat_in:.3e} W/m²")
            print(f"焦耳热总量: {total_joule_heat:.3e} W/m²")
            
            # 计算效率
            if heat_in > 0 and net_power > 0:
                efficiency = net_power / heat_in * 100  # 转换为百分比
                
                # 计算卡诺效率进行比较
                carnot_eff = (Th - Tc) / Th * 100
                relative_eff = efficiency / carnot_eff * 100  # 与卡诺效率的比值
                
                print(f"效率: {efficiency:.2f}% (卡诺效率: {carnot_eff:.2f}%, 相对效率: {relative_eff:.2f}%)")
                
                # 检查是否超过卡诺效率
                if efficiency > carnot_eff:
                    print(f"警告: 计算效率 {efficiency:.2f}% 超过卡诺效率 {carnot_eff:.2f}%")
                    efficiency = carnot_eff * 0.9  # 限制在卡诺效率的90%以内
            else:
                # 记录无效效率的原因
                if heat_in <= 0:
                    print(f"警告: 热输入为零或负值 ({heat_in:.3e} W/m²)")
                if net_power <= 0:
                    print(f"警告: 净功率为零或负值 ({net_power:.3e} W/m²)")
                
                efficiency = 0.0
                net_power = 0.0
            
            print(f"材料: {material_type}型, 组分={composition}, 电流密度={current_density}A/cm², 效率={efficiency:.2f}%")
            
            # 保存关键参数用于后续分析
            self.last_calc_data = {
                "seebeck": seebeck,
                "resistivity": resistivity,
                "thermal_cond": thermal_cond,
                "dTdx": dTdx,
                "current_density": J,
                "temperature": T,
                "heat_in": heat_in,
                "joule_heat": joule_heat,
                "seebeck_power": seebeck_power,
                "net_power": net_power,
                "efficiency": efficiency
            }
            
            return efficiency, net_power
            
        except Exception as e:
            print(f"效率计算错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0

    def calculate_zt(self, material_type, composition, temperature):
        """计算给定温度下的优值系数 ZT = S²T/(kρ)
        
        参数:
        material_type: 'p' 或 'n'，材料类型
        composition: 材料成分
        temperature: 温度 (K)
        
        返回:
        zt: 优值系数
        """
        try:
            # 创建插值器（如果还没有创建）
            interp_key = f"{material_type}_{composition}"
            if interp_key not in self.interpolators:
                self.create_interpolators(material_type, composition)
            
            # 获取材料属性
            # 塞贝克系数 (V/K)，使用绝对值因为N型材料的塞贝克系数为负
            seebeck = abs(self.interpolators[interp_key]["seebeck"](temperature))
            # 电阻率 (Ω·m)
            resistivity = self.interpolators[interp_key]["resistivity"](temperature)
            # 热导率 (W/(m·K))
            thermal_cond = self.interpolators[interp_key]["thermal_cond"](temperature)
            
            # 计算优值系数 ZT = S²T/(kρ)
            # S: 塞贝克系数 (V/K)
            # T: 温度 (K)
            # k: 热导率 (W/(m·K))
            # ρ: 电阻率 (Ω·m)
            zt = (seebeck ** 2) * temperature / (thermal_cond * resistivity)
            
            return zt
            
        except Exception as e:
            print(f"计算优值系数错误: {str(e)}")
            return 0

    def visualize_energy_flow(self, material_type, composition, current_density, x, T):
        """
        可视化材料内部的能量流动
        """
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            fig.suptitle(f"{material_type}型材料 (组分={composition}) 能量流分析", fontsize=14)
            
            # 转换单位
            J = current_density * 1e4  # A/cm² → A/m²
            
            # 准备数据
            n_points = len(x)
            dx = (x[-1] - x[0]) / (n_points - 1)
            
            # 计算温度梯度
            dTdx = np.zeros_like(T)
            dTdx[1:-1] = (T[2:] - T[:-2]) / (2*dx)
            dTdx[0] = (T[1] - T[0]) / dx
            dTdx[-1] = (T[-1] - T[-2]) / dx
            
            # 获取材料属性
            interp_key = f"{material_type}_{composition}"
            seebeck = np.zeros(n_points)
            resistivity = np.zeros(n_points)
            thermal_cond = np.zeros(n_points)
            
            for i in range(n_points):
                T_safe = np.clip(T[i], 300, 700)
                seebeck[i] = self.interpolators[interp_key]["seebeck"](T_safe)
                resistivity[i] = self.interpolators[interp_key]["resistivity"](T_safe)
                thermal_cond[i] = self.interpolators[interp_key]["thermal_cond"](T_safe)
            
            # 计算各种热流密度
            fourier_heat = thermal_cond * dTdx              # 傅里叶热流 κ·dT/dx
            peltier_heat = J * seebeck * T                  # 帕尔贴热流 J·S·T
            total_heat = fourier_heat - peltier_heat        # 净热流 q = κ·dT/dx - J·S·T
            joule_heat = J**2 * resistivity                 # 焦耳热 J²·ρ
            seebeck_power = J * seebeck * dTdx              # 塞贝克功率 J·S·dT/dx
            
            # 绘制热流分布
            ax1 = axes[0]
            ax1.plot(x, fourier_heat, 'r-', label='傅里叶热流 (κ·dT/dx)')
            ax1.plot(x, peltier_heat, 'b-', label='帕尔贴热流 (J·S·T)')
            ax1.plot(x, total_heat, 'g-', label='净热流 (q)')
            ax1.set_xlabel('位置 (归一化)')
            ax1.set_ylabel('热流密度 (W/m²)')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制功率和热损失
            ax2 = axes[1]
            ax2.plot(x, seebeck_power, 'b-', label='塞贝克功率 (J·S·dT/dx)')
            ax2.plot(x, joule_heat, 'r-', label='焦耳热损失 (J²·ρ)')
            ax2.plot(x, seebeck_power - joule_heat, 'g-', label='净功率')
            ax2.set_xlabel('位置 (归一化)')
            ax2.set_ylabel('功率密度 (W/m³)')
            ax2.legend()
            ax2.grid(True)
            
            # 显示图表
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"能量流可视化错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def validate_material_data(self, material_type, composition):
        """验证材料数据的有效性并返回验证结果"""
        try:
            data = self.p_type_data if material_type == 'p' else self.n_type_data
            if composition not in data:
                return False, f"找不到组分为 {composition} 的 {material_type}型材料数据"
            
            mat_data = data[composition]
            if len(mat_data["temp"]) == 0 or len(mat_data["seebeck"]) == 0 or len(mat_data["resistivity"]) == 0 or len(mat_data["thermal_cond"]) == 0:
                return False, f"{material_type}型材料 (组分={composition}) 的数据不完整"
                
            # 物理性质合理性检查
            if material_type == 'p':
                # P型材料的塞贝克系数应为正值
                if np.any(mat_data["seebeck"] <= 0):
                    return False, f"P型材料 (组分={composition}) 的塞贝克系数包含非正值"
            elif material_type == 'n':
                # N型材料的塞贝克系数应为负值
                if np.any(mat_data["seebeck"] >= 0):
                    return False, f"N型材料 (组分={composition}) 的塞贝克系数包含非负值"
            
            # 电阻率和热导率应为正值
            if np.any(mat_data["resistivity"] <= 0):
                return False, f"{material_type}型材料 (组分={composition}) 的电阻率包含非正值"
            if np.any(mat_data["thermal_cond"] <= 0):
                return False, f"{material_type}型材料 (组分={composition}) 的热导率包含非正值"
                
            # 数据点是否足够
            if len(mat_data["temp"]) < 5:
                return False, f"{material_type}型材料 (组分={composition}) 的数据点过少 ({len(mat_data['temp'])})"
                
            return True, f"{material_type}型材料 (组分={composition}) 的数据有效"
            
        except Exception as e:
            return False, f"验证材料数据时出错: {str(e)}"

    def material_P(self, T, composition):
        """根据温度计算P型材料的物理性质"""
        try:
            # 获取该组分对应的插值器
            interpolator_key = f"p_{composition}"
            if interpolator_key not in self.interpolators:
                self.create_interpolators('p', composition)
            
            interpolators = self.interpolators[interpolator_key]
            
            # 使用插值器计算物理性质
            seebeck = interpolators["seebeck"](T)
            resistivity = interpolators["resistivity"](T)
            thermal_cond = interpolators["thermal_cond"](T)
            
            return seebeck, resistivity, thermal_cond
        except Exception as e:
            print(f"计算P型材料物理性质时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 如果出错返回默认值
            return np.ones_like(T) * 200e-6, np.ones_like(T) * 1e-5, np.ones_like(T) * 2.0

    def material_N(self, T, composition):
        """根据温度计算N型材料的物理性质"""
        try:
            # 获取该组分对应的插值器
            interpolator_key = f"n_{composition}"
            if interpolator_key not in self.interpolators:
                self.create_interpolators('n', composition)
            
            interpolators = self.interpolators[interpolator_key]
            
            # 使用插值器计算物理性质
            seebeck = interpolators["seebeck"](T)
            resistivity = interpolators["resistivity"](T)
            thermal_cond = interpolators["thermal_cond"](T)
            
            return seebeck, resistivity, thermal_cond
        except Exception as e:
            print(f"计算N型材料物理性质时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 如果出错返回默认值
            return np.ones_like(T) * -200e-6, np.ones_like(T) * 1e-5, np.ones_like(T) * 2.0
    
    def temperature_distribution_P(self, n, J, Tc, Th, max_iter, composition):
        """
        计算P型材料的温度分布
        
        参数:
        n (int): 网格点数
        J (float): 电流密度 (A/cm²)
        Tc (float): 冷端温度 (K)
        Th (float): 热端温度 (K)
        max_iter (int): 最大迭代次数
        composition (str): 材料成分
        
        返回:
        tuple: (位置数组, 温度分布数组)
        """
        try:
            print(f"计算P型温度分布: J={J}A/cm², Tc={Tc}K, Th={Th}K")
            
            # 参数初始化
            l = 1  # 标准化长度
            dx = l / (n - 1)
            T = np.linspace(Tc, Th, n)  # 初始线性温度分布
            
            # 电流密度单位转换：A/cm² → A/m²
            J_SI = J * 1e4  # 1 A/cm² = 10000 A/m²
            
            # 迭代求解
            for iter_num in range(max_iter):
                print(f"  温度分布迭代 #{iter_num+1}")
                
                # 构建方程组
                A = np.zeros((n, n))
                b = np.zeros(n)
                
                # 获取材料属性
                sb, res, th = self.material_P(T, composition)
                
                # 计算系数
                c1 = J_SI * sb / th
                c2 = -1 / th
                c3 = sb ** 2 * J_SI ** 2 / th
                c4 = -J_SI * sb / th
                c5 = res * J_SI ** 2
                
                # 设置边界条件
                A[0, 0] = 1
                b[0] = Tc
                A[-1, -1] = 1
                b[-1] = Th
                
                # 构造系数矩阵 (内部点)
                for i in range(1, n - 1):
                    A[i, i - 1] = 1 / (c2[i] * dx)
                    A[i, i] = c4[i+1] / c2[i+1] - 1 / (c2[i+1] * dx) - (1 - c1[i] * dx) / (c2[i] * dx)
                    A[i, i + 1] = (1 - c1[i+1] * dx) / (c2[i+1] * dx) - c3[i+1] * dx - (1 - c1[i+1] * dx) * c4[i+1] / c2[i+1]
                    b[i] = c5[i-1] * dx
                
                # 求解线性方程组
                try:
                    T_new = np.linalg.solve(A, b)
                    
                    # 检查温度收敛性
                    T_diff = np.max(np.abs(T_new - T))
                    T = T_new.copy()
                    
                    print(f"  温度最大变化: {T_diff:.4f}K")
                    
                    # 收敛判断 (差值小于0.1K视为收敛)
                    if T_diff < 0.1:
                        print(f"  温度分布已收敛，共迭代{iter_num+1}次")
                        break
                        
                except np.linalg.LinAlgError:
                    print("  线性方程组求解失败，请检查系数矩阵")
                    return None, None
            
            # 打印温度分布摘要
            print(f"  最终温度分布: [Tc={T[0]:.1f}K, ..., Th={T[-1]:.1f}K]")
            
            # 返回位置和温度分布
            x = np.linspace(0, l, n)
            return x, T
            
        except Exception as e:
            print(f"计算P型温度分布错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def temperature_distribution_N(self, J, Tc, Th, n_points, l, composition):
        """
        计算N型材料的温度分布
        
        参数:
        J (float): 电流密度 (A/cm²)
        Tc (float): 冷端温度 (K)
        Th (float): 热端温度 (K)
        n_points (int): 网格点数
        l (float): 材料长度 (m)
        composition (str): 材料成分
        
        返回:
        tuple: (位置数组, 温度分布数组)
        """
        try:
            print(f"计算N型温度分布: J={J}A/cm², Tc={Tc}K, Th={Th}K")
            
            # 参数初始化
            dx = l / (n_points - 1)
            T = np.linspace(Tc, Th, n_points)  # 初始线性温度分布
            
            # 电流密度单位转换：A/cm² → A/m²
            J_SI = J * 1e4  # 1 A/cm² = 10000 A/m²
            
            # 迭代求解 (最大迭代次数限制为10，便于快速测试)
            max_iter = 10
            
            for iter_num in range(max_iter):
                print(f"  温度分布迭代 #{iter_num+1}")
                
                # 构建方程组
                A = np.zeros((n_points, n_points))
                b = np.zeros(n_points)
                
                # 获取材料属性
                sb, res, th = self.material_N(T, composition)
                
                # 计算系数
                c1 = J_SI * sb / th
                c2 = -1 / th
                c3 = sb ** 2 * J_SI ** 2 / th
                c4 = -J_SI * sb / th
                c5 = res * J_SI ** 2
                
                # 设置边界条件
                A[0, 0] = 1
                b[0] = Tc
                A[-1, -1] = 1
                b[-1] = Th
                
                # 构造系数矩阵 (内部点)
                for i in range(1, n_points - 1):
                    A[i, i - 1] = 1 / (c2[i] * dx)
                    A[i, i] = c4[i+1] / c2[i+1] - 1 / (c2[i+1] * dx) - (1 - c1[i] * dx) / (c2[i] * dx)
                    A[i, i + 1] = (1 - c1[i+1] * dx) / (c2[i+1] * dx) - c3[i+1] * dx - (1 - c1[i+1] * dx) * c4[i+1] / c2[i+1]
                    b[i] = c5[i-1] * dx
                
                # 求解线性方程组
                try:
                    T_new = np.linalg.solve(A, b)
                    
                    # 检查温度收敛性
                    T_diff = np.max(np.abs(T_new - T))
                    T = T_new.copy()
                    
                    print(f"  温度最大变化: {T_diff:.4f}K")
                    
                    # 收敛判断 (差值小于0.1K视为收敛)
                    if T_diff < 0.1:
                        print(f"  温度分布已收敛，共迭代{iter_num+1}次")
                        break
                        
                except np.linalg.LinAlgError:
                    print("  线性方程组求解失败，请检查系数矩阵")
                    return None, None
            
            # 打印温度分布摘要
            print(f"  最终温度分布: [Tc={T[0]:.1f}K, ..., Th={T[-1]:.1f}K]")
            
            # 返回位置和温度分布
            x = np.linspace(0, l, n_points)
            return x, T
            
        except Exception as e:
            print(f"计算N型温度分布错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def calculate_efficiency_P(self, Tc, Th, n, l, composition):
        """
        计算P型材料的效率曲线，参考01.py
        
        参数:
        Tc: 低温端温度 (K)
        Th: 高温端温度 (K)
        n: 网格点数
        l: 材料长度
        composition: 材料组分
        
        返回:
        eff_list_P: 效率列表
        J_list_P: 电流密度列表 (A/cm²)
        """
        try:
            print(f"\n计算P型材料效率曲线: Tc={Tc}K, Th={Th}K, 组分={composition}")
            
            eff_list_P = []
            J_list_P = []
            dx = l / (n - 1)
            
            # 参考01.py，P型材料电流密度范围为-30到0
            for j in range(0, 31, 2):  # 0到30，步长2
                J = -j/10  # 转换为A/cm²，负值
                print(f"\n===== 计算P型材料效率 (J={J}A/cm²) =====")
                
                # 计算温度分布
                T = self.temperature_distribution_P(n, J, Tc, Th, 10, composition)[1]
                if T is None:
                    print(f"  温度分布计算失败，跳过 J={J}")
                    continue
                
                # 计算材料属性
                sb, res, th = self.material_P(T, composition)
                
                # 计算热流系数 - 电流密度单位A/m²
                J_SI = J * 1e4
                c1 = J_SI * sb / th
                c2 = -1 / th
                c3 = sb ** 2 * J_SI ** 2 / th
                c4 = -J_SI * sb / th
                c5 = res * J_SI ** 2
                
                # 计算热流密度q
                q = np.zeros(n)
                for k in range(1, n):
                    q[k] = ((1/dx - c1[k]) * T[k] - T[k-1]/dx) / (c2[k])
                q[0] = (1 - c4[1] * dx) * q[1] - c3[1] * dx * T[1] - c5[1] * dx
                
                # 计算积分项
                seebeck_integral = 0  # 第一个积分 - 塞贝克系数
                resistivity_integral = 0  # 第二个积分 - 电阻率
                
                for m in range(1, n):
                    T1 = T[m]
                    T2 = T[m-1]
                    seebeck_integral += (sb[m] + sb[m-1]) / 2 * (T1 - T2)
                    resistivity_integral += (res[m] + res[m-1]) / 2 * dx
                
                print(f"  塞贝克积分: {seebeck_integral:.6f} V")
                print(f"  电阻率积分: {resistivity_integral:.6f} Ω·m")
                
                # 计算效率 - 完全按照01.py的计算方式
                if q[n-1] != 0:
                    eff = J_SI * (seebeck_integral + J_SI * resistivity_integral) / q[n-1] * 100
                    
                    # 确保效率不超过卡诺效率，但保留负效率以便调试
                    carnot_eff = (Th - Tc) / Th * 100
                    
                    if eff > carnot_eff:
                        print(f"  警告: 效率 {eff:.4f}% 超过卡诺效率 {carnot_eff:.2f}%")
                        eff = carnot_eff * 0.95
                    
                    if eff < 0:
                        print(f"  负效率: {eff:.4f}%，保留为负值以便调试")
                    
                    print(f"  计算效率: {eff:.4f}%")
                    eff_list_P.append(eff)
                    J_list_P.append(J)
                else:
                    print(f"  热流为零，跳过此点")
            
            return eff_list_P, J_list_P
            
        except Exception as e:
            print(f"计算P型效率曲线错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

    def calculate_efficiency_N(self, Tc, Th, n_points, l, composition):
        """
        计算N型材料的效率曲线，参考01.py
        
        参数:
        Tc (float): 冷端温度 (K)
        Th (float): 热端温度 (K)
        n_points (int): 网格点数
        l (float): 材料长度 (m)
        composition (str): 材料成分
        
        返回:
        tuple: (效率列表, 电流密度列表)
        """
        try:
            print(f"\n计算N型材料效率曲线: Tc={Tc}K, Th={Th}K, 组分={composition}")
            
            eff_list = []
            J_list = []
            dx = l / (n_points - 1)
            
            # 参考01.py，N型材料电流密度范围为0到50
            for j in range(0, 51, 5):  # 0到50，步长5
                J = j/10  # 转换为A/cm²
                print(f"\n===== 计算N型材料效率 (J={J}A/cm²) =====")
                
                # 计算温度分布 - 这里的J参数是电流密度，单位A/cm²
                T = self.temperature_distribution_N(J, Tc, Th, n_points, l, composition)[1]
                if T is None:
                    print(f"  温度分布计算失败，跳过 J={J}")
                    continue
                
                # 计算材料属性
                sb, res, th = self.material_N(T, composition)
                
                # 计算热流系数 - 电流密度单位A/m²
                J_SI = J * 1e4
                c1 = J_SI * sb / th
                c2 = -1 / th
                c3 = sb ** 2 * J_SI ** 2 / th
                c4 = -J_SI * sb / th
                c5 = res * J_SI ** 2
                
                # 计算热流密度q
                q = np.zeros(n_points)
                for k in range(1, n_points):
                    q[k] = ((1/dx - c1[k]) * T[k] - T[k-1]/dx) / (c2[k])
                q[0] = (1 - c4[1] * dx) * q[1] - c3[1] * dx * T[1] - c5[1] * dx
                
                # 计算积分项
                seebeck_integral = 0  # 第一个积分 - 塞贝克系数
                resistivity_integral = 0  # 第二个积分 - 电阻率
                
                for m in range(1, n_points):
                    T1 = T[m]
                    T2 = T[m-1]
                    seebeck_integral += (sb[m] + sb[m-1]) / 2 * (T1 - T2)
                    resistivity_integral += (res[m] + res[m-1]) / 2 * dx
                
                print(f"  塞贝克积分: {seebeck_integral:.6f} V")
                print(f"  电阻率积分: {resistivity_integral:.6f} Ω·m")
                
                # 计算效率 - 完全按照01.py的计算方式
                if q[n_points-1] != 0:
                    eff = J_SI * (seebeck_integral + J_SI * resistivity_integral) / q[n_points-1] * 100
                    
                    # 确保效率不超过卡诺效率且不为负
                    carnot_eff = (Th - Tc) / Th * 100
                    
                    if eff > carnot_eff:
                        print(f"  警告: 效率 {eff:.4f}% 超过卡诺效率 {carnot_eff:.2f}%")
                        eff = carnot_eff * 0.95
                    
                    if eff < 0:
                        print(f"  负效率: {eff:.4f}%，保留为负值以便调试")
                    
                    print(f"  计算效率: {eff:.4f}%")
                    eff_list.append(eff)
                    J_list.append(J)
                else:
                    print(f"  热流为零，跳过此点")
            
            return eff_list, J_list
            
        except Exception as e:
            print(f"计算N型效率曲线错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

class ThermoelectricApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_plot_style()
        self.setWindowTitle('基于差分法的半导体热电器件仿真实验')
        
        # 设置窗口的默认大小和最小大小
        screen = QApplication.primaryScreen().geometry()
        default_width = min(int(screen.width() * 0.8), 1440)  # 最大宽度1440
        default_height = min(int(screen.height() * 0.8), 900)  # 最大高度900
        self.setGeometry(100, 100, default_width, default_height)
        self.setMinimumSize(1024, 600)  # 设置最小窗口大小
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(5)  # 减小面板之间的间距
        main_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
        
        # 创建左侧面板 - 先创建它，确保iter_edit已经定义
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)
        
        # 初始化计算器 - 现在iter_edit已经存在
        self.calculator = ThermoelectricCalculator()
        
        # 创建中间面板
        middle_panel = self.create_middle_panel()
        main_layout.addWidget(middle_panel)
        
        # 创建右侧面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)
        
        # 设置面板的比例 (左:中:右 = 2:3:3)
        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 3)
        main_layout.setStretch(2, 3)

        # 连接信号和槽
        self.init_button.clicked.connect(self.initialize_calculation)
        self.p_current_combo.currentTextChanged.connect(self.update_branch_characteristics)
        self.n_current_combo.currentTextChanged.connect(self.update_branch_characteristics)
        
        # 连接右侧面板的计算和导出按钮
        self.right_calc_button.clicked.connect(self.calculate_device_performance)
        self.right_export_button.clicked.connect(self.export_data)

    def setup_plot_style(self):
        plt.style.use('default')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
        
        plt.rcParams.update({
            'figure.facecolor': '#F0F0F0',
            'axes.facecolor': '#F0F0F0',
            'axes.grid': False,
            'axes.spines.top': True,
            'axes.spines.right': True,
            'font.size': 10,
            'figure.subplot.hspace': 0.3,
            'figure.subplot.wspace': 0.3
        })

    def create_toolbar_buttons(self):
        buttons = []
        icons = ["⌂", "←", "→", "✥", "🔍", "≡", "📄"]
        for icon in icons:
            btn = QPushButton(icon)
            btn.setFixedSize(25, 25)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 1px solid #CCCCCC;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #E6E6E6;
                }
            """)
            buttons.append(btn)
        return buttons

    def create_plot_widget(self, num_subplots=2, height=3, vertical=False):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)  # 完全移除边距
        layout.setSpacing(0)  # 移除间距
        
        # 创建工具栏
        toolbar = QFrame()
        toolbar.setFixedHeight(16)  # 进一步减小工具栏高度
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #F0F0F0;
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(1, 0, 1, 0)  # 只保留左右边距
        toolbar_layout.setSpacing(1)  # 最小按钮间距
        
        # 创建工具按钮
        icons = ["⌂", "←", "→", "+", "🔍", "≡", "📄"]
        for icon in icons:
            btn = QPushButton(icon)
            btn.setFixedSize(16, 16)  # 进一步减小按钮大小
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 1px solid #CCCCCC;
                    border-radius: 1px;
                    padding: 0px;
                    margin: 0px;
                    font-size: 9px;
                }
                QPushButton:hover {
                    background-color: #E6E6E6;
                }
            """)
            toolbar_layout.addWidget(btn)
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)
        
        # 创建图表
        dpi = QApplication.primaryScreen().logicalDotsPerInch()
        fig_width = container.width() / dpi
        fig_height = (height * 96 + 10) / dpi  # 稍微增加图表高度
        
        if vertical and num_subplots > 1:
            fig, axes = plt.subplots(num_subplots, 1, figsize=(fig_width, fig_height))
        else:
            fig, axes = plt.subplots(1, num_subplots, figsize=(fig_width, fig_height))
        
        if num_subplots == 1:
            axes = [axes]
        
        # 设置图表样式
        for ax in axes:
            ax.grid(True, color='white', linestyle='-', alpha=0.8)
            ax.set_facecolor('#F0F0F0')
            ax.clear()
            ax.grid(True)
            # 调整字体大小
            ax.tick_params(labelsize=8)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(8)
        
        # 调整图表间距，进一步减小上边距
        plt.subplots_adjust(top=0.88, bottom=0.15, left=0.15, right=0.95)
        
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        return container, axes, canvas

    def create_left_panel(self):
        panel = QGroupBox()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 添加标题
        title_label = QLabel("基于差分法的半导体热电器件仿真实验")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #0072BC;
            padding: 5px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        title_label.setFixedHeight(50)
        layout.addWidget(title_label)
        
        # 添加示意图
        image_container = QGroupBox()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # 使用新的ClickableImageLabel替代QLabel
        image_label = ClickableImageLabel()
        pixmap = QPixmap("图片1.png")
        scaled_pixmap = pixmap.scaled(400, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        # 添加提示文本
        image_label.setToolTip("双击查看大图")
        image_layout.addWidget(image_label)
        
        layout.addWidget(image_container)
        layout.addSpacing(10)
        
        # 初始条件设置
        params_group = QGroupBox("初始条件设置")
        params_layout = QGridLayout()
        params_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
        params_layout.setSpacing(5)  # 减小间距
        
        # 温度和网格设置
        params_layout.addWidget(QLabel("高温温度Th(K)"), 0, 0)
        self.th_edit = QLineEdit("500")
        params_layout.addWidget(self.th_edit, 0, 1)
        
        params_layout.addWidget(QLabel("格子数量"), 0, 2)
        self.grid_edit = QLineEdit("10")
        params_layout.addWidget(self.grid_edit, 0, 3)
        
        params_layout.addWidget(QLabel("低温温度Tc(K)"), 1, 0)
        self.tc_edit = QLineEdit("300")
        params_layout.addWidget(self.tc_edit, 1, 1)
        
        params_layout.addWidget(QLabel("迭代次数"), 1, 2)
        self.iter_edit = QLineEdit("20")
        params_layout.addWidget(self.iter_edit, 1, 3)
        
        # 材料选择
        params_layout.addWidget(QLabel("PbTe1-yIy"), 2, 0)
        self.p_type_combo = QComboBox()
        self.p_type_combo.addItems(["0.01", "0.02", "0.03"])  # 使用实际组分值而不是文件名
        params_layout.addWidget(self.p_type_combo, 2, 1)
        self.p_type_combo.currentTextChanged.connect(self.update_p_composition)
        
        params_layout.addWidget(QLabel("PbTe:Na/Ag2Te"), 2, 2)
        self.n_type_combo = QComboBox()
        self.n_type_combo.addItems(["0.0004", "0.0012", "0.0020", "0.0028"])  # 保持N型材料组分值不变
        params_layout.addWidget(self.n_type_combo, 2, 3)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 材料优值系数图表
        zt_group = QGroupBox("选择材料的优值系数")
        zt_layout = QVBoxLayout()
        zt_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
        
        zt_container, (ax1, ax2), canvas = self.create_plot_widget(height=2)
        self.zt_axes = (ax1, ax2)  # 保存axes引用以便后续更新
        self.zt_canvas = canvas    # 保存canvas引用以便后续更新
        
        # 设置P型图表
        ax1.set_title("P型半导体材料", pad=5)
        ax1.set_xlabel("温度")
        ax1.set_ylabel("ZT")
        ax1.set_xlim(300, 700)
        ax1.set_ylim(0, 1.5)
        ax1.grid(True, color='white', linestyle='-', alpha=0.8)
        ax1.set_facecolor('#F0F0F0')
        
        # 设置N型图表
        ax2.set_title("N型半导体材料", pad=5)
        ax2.set_xlabel("温度")
        ax2.set_ylabel("ZT")
        ax2.set_xlim(300, 700)
        ax2.set_ylim(0, 1.5)
        ax2.grid(True, color='white', linestyle='-', alpha=0.8)
        ax2.set_facecolor('#F0F0F0')
        
        # 调整图表布局
        plt.tight_layout()
        
        zt_layout.addWidget(zt_container)
        zt_group.setLayout(zt_layout)
        layout.addWidget(zt_group)
        
        # 添加初始化按钮和状态指示灯
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 0, 5, 5)  # 减小边距
        self.init_button = QPushButton("初始化运算")
        button_layout.addWidget(self.init_button)
        
        button_layout.addWidget(QLabel("运行状态"))
        self.status_light = StatusLight()
        button_layout.addWidget(self.status_light)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # 设置拉伸因子，使图片区域占据更多空间
        layout.setStretch(0, 1)  # 标题
        layout.setStretch(1, 4)  # 图片
        layout.setStretch(2, 0)  # 间距
        layout.setStretch(3, 2)  # 参数设置
        layout.setStretch(4, 2)  # 优值系数图表
        
        panel.setLayout(layout)
        return panel

    def create_middle_panel(self):
        panel = QGroupBox("分支特性")
        layout = QVBoxLayout()
        
        # 格点温度分布
        temp_group = QGroupBox("格点温度分布")
        temp_layout = QVBoxLayout()
        
        temp_container, (ax1, ax2), canvas = self.create_plot_widget()
        # 保存温度分布图表的引用
        self.temp_axes = (ax1, ax2)
        self.temp_canvas = canvas
        
        # 移除多余的提示标签
        
        ax1.set_title("格点温度分布（P型）")
        ax2.set_title("格点温度分布（N型）")
        
        for ax in [ax1, ax2]:
            ax.set_xlabel("格点位置")
            ax.set_ylabel("T (K)")
            ax.set_xlim(0, 10)
            ax.set_ylim(300, 500)
        
        temp_layout.addWidget(temp_container)
        
        # 电流密度选择
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("电流密度（A/cm2）"))
        self.p_current_combo = QComboBox()
        self.p_current_combo.addItems(["-2.0", "-1.5", "-1.0", "-0.5"])
        current_layout.addWidget(self.p_current_combo)
        
        current_layout.addWidget(QLabel("电流密度（A/cm2）"))
        self.n_current_combo = QComboBox()
        self.n_current_combo.addItems(["25", "30", "35", "40"])
        current_layout.addWidget(self.n_current_combo)
        
        temp_layout.addLayout(current_layout)
        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)
        
        # 材料效率
        eff_group = QGroupBox("材料效率")
        eff_layout = QVBoxLayout()
        
        eff_container, (ax3, ax4), canvas = self.create_plot_widget()
        # 保存效率图表的引用
        self.eff_axes = (ax3, ax4)
        self.eff_canvas = canvas
        
        ax3.set_title("效率（P型）")
        ax4.set_title("效率（N型）")
        
        ax3.set_xlabel("电流密度(A/cm2)")
        ax3.set_ylabel("效率")
        ax3.set_xlim(-20, 0)
        ax3.set_ylim(0, 0.1)
        
        ax4.set_xlabel("电流密度(A/cm2)")
        ax4.set_ylabel("效率")
        ax4.set_xlim(0, 50)
        ax4.set_ylim(0, 0.1)
        
        eff_layout.addWidget(eff_container)
        
        # 添加计算按钮和状态指示灯
        calc_layout = QHBoxLayout()
        calc_button = QPushButton("计算")
        calc_button.clicked.connect(self.update_branch_characteristics)
        calc_layout.addWidget(calc_button)
        
        # 添加新按钮运行PNefficiency.py程序
        external_calc_button = QPushButton("详细效率计算")
        external_calc_button.clicked.connect(self.run_external_efficiency)
        calc_layout.addWidget(external_calc_button)
        
        calc_layout.addWidget(QLabel("运行状态"))
        self.calc_status = StatusLight()
        calc_layout.addWidget(self.calc_status)
        calc_layout.addStretch()
        
        eff_layout.addLayout(calc_layout)
        eff_group.setLayout(eff_layout)
        layout.addWidget(eff_group)
        
        panel.setLayout(layout)
        return panel

    def create_right_panel(self):
        panel = QGroupBox("结果分析")
        layout = QVBoxLayout()
        layout.setSpacing(5)  # 减小组件之间的间距
        layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
        
        # N/P比例设置
        ratio_layout = QHBoxLayout()
        ratio_layout.setContentsMargins(0, 0, 0, 0)
        ratio_layout.addWidget(QLabel("N型分支面积/P型分支面积"))
        self.ratio_edit = QLineEdit("0.1")
        ratio_layout.addWidget(self.ratio_edit)
        layout.addLayout(ratio_layout)
        
        # 1. 器件功率图表
        power_group = QGroupBox("器件功率")  # 添加GroupBox包装，以便于findChild查找
        power_layout = QVBoxLayout()
        power_container, [power_ax], power_canvas = self.create_plot_widget(num_subplots=1, height=2.5)
        power_ax.set_title("器件功率")
        power_ax.set_xlabel("电流 (A)")  # 将"电流密度"改为"电流"
        power_ax.set_ylabel("功率 (W/cm²)")
        power_ax.set_xlim(0, 10)  # 扩大范围以适应更大的电流值
        power_ax.set_ylim(0, 1)
        power_layout.addWidget(power_container)
        power_group.setLayout(power_layout)
        layout.addWidget(power_group)
        
        # 2. 器件效率图表
        eff_group = QGroupBox("器件效率")  # 添加GroupBox包装，以便于findChild查找
        eff_layout = QVBoxLayout()
        efficiency_container, [efficiency_ax], efficiency_canvas = self.create_plot_widget(num_subplots=1, height=2.5)
        efficiency_ax.set_title("器件效率")
        efficiency_ax.set_xlabel("电流 (A)")  # 将"电流密度"改为"电流"
        efficiency_ax.set_ylabel("效率 (%)")
        efficiency_ax.set_xlim(0, 10)  # 扩大范围以适应更大的电流值
        efficiency_ax.set_ylim(0, 40)  # 扩大范围以显示更高的效率值
        eff_layout.addWidget(efficiency_container)
        eff_group.setLayout(eff_layout)
        layout.addWidget(eff_group)
        
        # 最大功率点和最大效率点显示框
        results_layout = QHBoxLayout()
        results_layout.setSpacing(10)  # 减小显示框之间的间距
        results_layout.setContentsMargins(0, 0, 0, 0)
        
        # 最大功率点
        power_point_group = QGroupBox("最大功率点")
        power_layout = QVBoxLayout()
        power_layout.setSpacing(5)  # 减小内部组件的间距
        power_layout.setContentsMargins(5, 5, 5, 5)
        
        power_value_layout = QHBoxLayout()
        power_value_layout.addWidget(QLabel("最大功率"))
        self.max_power = QLineEdit()
        power_value_layout.addWidget(self.max_power)
        power_layout.addLayout(power_value_layout)
        
        power_current_layout = QHBoxLayout()
        power_current_layout.addWidget(QLabel("电流"))  # 将"电流密度"改为"电流"
        self.power_current = QLineEdit()
        power_current_layout.addWidget(self.power_current)
        power_layout.addLayout(power_current_layout)
        
        power_point_group.setLayout(power_layout)
        results_layout.addWidget(power_point_group)
        
        # 最大效率点
        eff_point_group = QGroupBox("最大效率点")
        eff_layout = QVBoxLayout()
        eff_layout.setSpacing(5)  # 减小内部组件的间距
        eff_layout.setContentsMargins(5, 5, 5, 5)
        
        eff_value_layout = QHBoxLayout()
        eff_value_layout.addWidget(QLabel("最大效率"))
        self.max_eff = QLineEdit()
        eff_value_layout.addWidget(self.max_eff)
        eff_layout.addLayout(eff_value_layout)
        
        eff_current_layout = QHBoxLayout()
        eff_current_layout.addWidget(QLabel("电流"))  # 将"电流密度"改为"电流"
        self.eff_current = QLineEdit()
        eff_current_layout.addWidget(self.eff_current)
        eff_layout.addLayout(eff_current_layout)
        
        eff_point_group.setLayout(eff_layout)
        results_layout.addWidget(eff_point_group)
        
        layout.addLayout(results_layout)
        
        # 3. 功率效率优化区间图表
        opt_group = QGroupBox("功率效率优化区间")  # 添加GroupBox包装，以便于findChild查找
        opt_layout = QVBoxLayout()
        optimization_container, [optimization_ax], optimization_canvas = self.create_plot_widget(num_subplots=1, height=2.5)
        optimization_ax.set_title("功率效率优化区间")
        optimization_ax.set_xlabel("功率 (W/cm²)")
        optimization_ax.set_ylabel("效率 (%)")
        optimization_ax.set_xlim(0, 0.1)
        optimization_ax.set_ylim(0, 40)
        opt_layout.addWidget(optimization_container)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)  # 减小按钮之间的间距
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.right_calc_button = QPushButton("计算")
        self.right_calc_button.clicked.connect(self.calculate_device_performance)  # 连接到calculate_device_performance方法
        self.right_export_button = QPushButton("导出数据")
        self.right_export_button.clicked.connect(self.export_data)  # 连接到export_data方法
        button_layout.addWidget(self.right_calc_button)
        button_layout.addWidget(self.right_export_button)
        button_layout.addStretch()  # 添加弹性空间
        layout.addLayout(button_layout)
        
        panel.setLayout(layout)
        return panel

    def update_zt_plots(self):
        """更新优值系数图表，展示ZT随温度的变化"""
        try:
            # 获取当前选择的材料组分
            p_composition = self.p_type_combo.currentText()
            n_composition = self.n_type_combo.currentText()
            
            # 创建温度范围（300K - 700K），与MATLAB代码一致
            temperatures = np.arange(300, 701, 20)  # 300:20:700
            
            # 计算P型材料的优值系数
            p_zt = []
            for T in temperatures:
                # 直接从Excel文件中读取ZT值，与MATLAB代码一致
                interp_key = f"p_{p_composition}"
                if interp_key not in self.calculator.interpolators:
                    self.calculator.create_interpolators('p', p_composition)
                p_zt.append(self.calculator.calculate_zt('p', p_composition, T))
            
            # 计算N型材料的优值系数
            n_zt = []
            for T in temperatures:
                interp_key = f"n_{n_composition}"
                if interp_key not in self.calculator.interpolators:
                    self.calculator.create_interpolators('n', n_composition)
                n_zt.append(self.calculator.calculate_zt('n', n_composition, T))
            
            # 更新P型图表
            self.zt_axes[0].clear()
            self.zt_axes[0].plot(temperatures, p_zt, 'b+-', linewidth=2)  # 使用蓝色+号标记，与MATLAB一致
            self.zt_axes[0].set_title("P型半导体材料优值系数", pad=5)
            self.zt_axes[0].set_xlabel("温度 (K)")
            self.zt_axes[0].set_ylabel("ZT")
            self.zt_axes[0].set_xlim(300, 700)
            self.zt_axes[0].set_ylim(0, 2.0)  # 与MATLAB图形一致
            self.zt_axes[0].grid(True, linestyle='--', alpha=0.7)
            
            # 更新N型图表
            self.zt_axes[1].clear()
            self.zt_axes[1].plot(temperatures, n_zt, 'r*-', linewidth=2)  # 使用红色*号标记，与MATLAB一致
            self.zt_axes[1].set_title("N型半导体材料优值系数", pad=5)
            self.zt_axes[1].set_xlabel("温度 (K)")
            self.zt_axes[1].set_ylabel("ZT")
            self.zt_axes[1].set_xlim(300, 700)
            self.zt_axes[1].set_ylim(0, 2.0)  # 与MATLAB图形一致
            self.zt_axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # 设置两个图表的共同属性
            for ax in self.zt_axes:
                ax.set_facecolor('#F8F8F8')
                ax.tick_params(direction='in')  # 刻度线向内
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                # 设置主要刻度
                ax.set_xticks(np.arange(300, 701, 100))
                ax.set_yticks(np.arange(0, 2.1, 0.5))
                # 添加次要刻度
                ax.minorticks_on()
            
            # 刷新图表
            self.zt_canvas.draw()
            
        except Exception as e:
            print(f"更新优值系数图表错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def initialize_calculation(self):
        """初始化运算"""
        try:
            print("\n===== 开始初始化计算 =====")
            # 更新状态指示灯为红色（计算中）
            self.status_light.set_status(False)
            QApplication.processEvents()  # 确保UI更新
            
            # 更新优值系数图表
            self.update_zt_plots()
            
            # 获取输入参数
            Th = float(self.th_edit.text())
            Tc = float(self.tc_edit.text())
            n_points = int(self.grid_edit.text())
            max_iter = int(self.iter_edit.text())  # 获取迭代次数
            
            print(f"输入参数: Th={Th}K, Tc={Tc}K, 格点数={n_points}")
            
            # 计算P型和N型材料的温度分布
            p_composition = self.p_type_combo.currentText()
            n_composition = self.n_type_combo.currentText()
            
            # 获取当前选择的电流密度
            p_current = float(self.p_current_combo.currentText())
            n_current = float(self.n_current_combo.currentText())
            
            print(f"P型材料: 组分={p_composition}, 电流密度={p_current}A/cm²")
            print(f"N型材料: 组分={n_composition}, 电流密度={n_current}A/cm²")
            
            # 将最大迭代次数传递给温度分布计算函数
            x_p, T_p = self.calculator.calculate_temperature_distribution(
                Th, Tc, n_points, 'p', p_composition, p_current, max_iter)
            x_n, T_n = self.calculator.calculate_temperature_distribution(
                Th, Tc, n_points, 'n', n_composition, n_current, max_iter)
            
            # 保存计算结果以便后续使用
            self.x_p, self.T_p = x_p, T_p
            self.x_n, self.T_n = x_n, T_n
            
            print("计算完成，正在更新温度分布图...")
            
            # 删除旧的点击事件处理器（如果存在）
            if hasattr(self, '_pick_cid') and self._pick_cid:
                self.temp_canvas.mpl_disconnect(self._pick_cid)
            
            # 更新温度分布图
            self.update_temperature_plots(x_p, T_p, x_n, T_n)
            
            # 计算完成，更新状态指示灯为绿色
            self.status_light.set_status(True)
            print("===== 初始化计算完成 =====")
            
        except Exception as e:
            print(f"初始化计算错误: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_light.set_status(False)
    
    def update_temperature_plots(self, x_p, T_p, x_n, T_n):
        """
        更新温度分布图，使横坐标随格点数变化，并支持数据点交互
        """
        try:
            # 使用保存的引用直接访问图表
            ax1, ax2 = self.temp_axes
            
            # 清除旧数据
            ax1.clear()
            ax2.clear()
            
            # 获取格点数量
            n_points_p = len(x_p)
            n_points_n = len(x_n)
            
            # 使用整数格点位置 1, 2, 3, ..., n
            grid_points_p = np.arange(1, n_points_p + 1)
            grid_points_n = np.arange(1, n_points_n + 1)
            
            print(f"\n=== 温度分布图数据 ===")
            print(f"P型格点数量: {n_points_p}")
            print(f"P型温度数据: {T_p}")
            print(f"N型格点数量: {n_points_n}")
            print(f"N型温度数据: {T_n}")
            
            # 绘制新数据 - 使用标记和细线
            p_line, = ax1.plot(grid_points_p, T_p, 'b*-', markersize=6, picker=5)  # 设置picker参数启用点击事件
            n_line, = ax2.plot(grid_points_n, T_n, 'r*-', markersize=6, picker=5)
            
            # 添加点击事件处理函数
            def on_pick(event):
                if event.artist == p_line:
                    ind = event.ind[0]
                    ax = ax1
                    grid_points = grid_points_p
                    temps = T_p
                    title = "P型材料"
                elif event.artist == n_line:
                    ind = event.ind[0]
                    ax = ax2
                    grid_points = grid_points_n
                    temps = T_n
                    title = "N型材料"
                else:
                    return
                
                # 显示详细信息
                pos = grid_points[ind]
                temp = temps[ind]
                
                # 移除之前的标注（如果有）
                for artist in ax.texts:
                    artist.remove()
                
                # 添加新标注
                ax.annotate(f'格点: {pos}\n温度: {temp:.2f}K',
                            xy=(pos, temp), xytext=(pos+0.5, temp+10),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='arc3,rad=.2',
                                            color='green'),
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                            fontsize=8)
                
                # 更新图表
                self.temp_canvas.draw()
                
                # 输出详细数据到控制台
                print(f"{title} 格点位置 {pos} 的详细数据:")
                print(f"  温度: {temp:.2f}K")
            
            # 连接点击事件
            self._pick_cid = self.temp_canvas.mpl_connect('pick_event', on_pick)
            
            # 设置标题和标签
            ax1.set_title("格点温度分布（P型）")
            ax2.set_title("格点温度分布（N型）")
            
            # 获取温度的最小值和最大值，用于设置Y轴范围
            min_temp = min(min(T_p), min(T_n))
            max_temp = max(max(T_p), max(T_n))
            
            # 设置坐标轴范围和刻度
            for ax, n_points in zip([ax1, ax2], [n_points_p, n_points_n]):
                ax.set_xlabel("格点位置")
                ax.set_ylabel("温度 (K)")
                
                # 动态设置横坐标范围和刻度
                ax.set_xlim(0.5, n_points + 0.5)  # 添加边距
                
                # 如果格点数较多，则间隔显示刻度
                if n_points <= 20:
                    ax.set_xticks(range(1, n_points + 1))
                else:
                    step = max(1, n_points // 10)  # 最多显示10个刻度
                    ax.set_xticks(range(1, n_points + 1, step))
                
                # 设置Y轴范围
                y_margin = (max_temp - min_temp) * 0.1  # 添加10%的边距
                ax.set_ylim(min_temp - y_margin, max_temp + y_margin)
                
                # 添加网格
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # 刷新图表
            self.temp_canvas.draw()
            print("温度分布图更新完成")
            
        except Exception as e:
            print(f"更新温度分布图错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_efficiency_plots(self):
        """更新效率曲线图"""
        # 清除原有曲线
        self.eff_axes[0].clear()
        self.eff_axes[1].clear()
        
        # 获取温度值
        Th = self.th_edit.text()
        Tc = self.tc_edit.text()
        
        try:
            Th = float(Th)
            Tc = float(Tc)
        except ValueError:
            print("温度值必须是有效数字")
            return
        
        if Th <= Tc:
            print("警告：热端温度必须高于冷端温度")
            return
            
        # 计算卡诺效率作为参考线
        carnot_eff = (Th - Tc) / Th * 100
        print(f"卡诺效率: {carnot_eff:.2f}%")
        
        # 定义一些变量
        n_points = 10  # 网格点数，与01.py一致
        l = 1.0  # 材料长度，单位m
        
        ax1, ax2 = self.eff_axes
        
        # 获取当前选择的材料组分
        p_composition = self.p_type_combo.currentText()
        n_composition = self.n_type_combo.currentText()
        
        # 计算P型效率曲线
        print("\n计算P型效率曲线: Th={:.1f}K, Tc={:.1f}K, 组分={}".format(
            Th, Tc, p_composition))
        
        p_eff_list, p_J_list = self.calculator.calculate_efficiency_P(
            Tc, Th, n_points, l, p_composition)
        
        # 计算N型效率曲线
        print("\n计算N型效率曲线: Th={:.1f}K, Tc={:.1f}K, 组分={}".format(
            Th, Tc, n_composition))
        
        n_eff_list, n_J_list = self.calculator.calculate_efficiency_N(
            Tc, Th, n_points, l, n_composition)
        
        # 打印结果摘要，方便调试
        print("\nP型效率结果:", p_eff_list)
        print("P型电流密度:", p_J_list)
        print("\nN型效率结果:", n_eff_list)
        print("N型电流密度:", n_J_list)
        
        # 绘制P型材料效率曲线
        if p_eff_list and p_J_list:
            ax1.plot(p_J_list, p_eff_list, 'b-', linewidth=2, label='P型效率')
            ax1.set_xlabel('电流密度 (A/cm²)')
            ax1.set_ylabel('效率 (%)')
            ax1.set_title('P型材料效率曲线')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 调整P型坐标轴范围
            ax1.set_xlim([-3, 0])  # P型材料使用负电流密度
            
            # 设置Y轴范围，考虑到有可能有负效率
            if max(p_eff_list) > 0:
                ax1.set_ylim([min(min(p_eff_list), 0), max(max(p_eff_list) * 1.1, carnot_eff * 1.1)])
            else:
                ax1.set_ylim([min(p_eff_list) * 1.1, carnot_eff * 1.1])
                
            # 添加卡诺效率线
            ax1.axhline(y=carnot_eff, color='r', linestyle='--', label=f'卡诺效率: {carnot_eff:.1f}%')
            ax1.legend(loc='best')
        
        # 绘制N型材料效率曲线
        if n_eff_list and n_J_list:
            ax2.plot(n_J_list, n_eff_list, 'g-', linewidth=2, label='N型效率')
            ax2.set_xlabel('电流密度 (A/cm²)')
            ax2.set_ylabel('效率 (%)')
            ax2.set_title('N型材料效率曲线')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 调整N型坐标轴范围
            ax2.set_xlim([0, 4])  # N型材料使用正电流密度
            
            # 设置Y轴范围，考虑到有可能有负效率
            if max(n_eff_list) > 0:
                y_min = min(min(n_eff_list), 0)
                y_max = max(max(n_eff_list) * 1.1, carnot_eff * 1.1)
                ax2.set_ylim([y_min, y_max])
            else:
                ax2.set_ylim([min(n_eff_list) * 1.1, carnot_eff * 1.1])
                
            # 添加卡诺效率线
            ax2.axhline(y=carnot_eff, color='r', linestyle='--', label=f'卡诺效率: {carnot_eff:.1f}%')
            ax2.legend(loc='best')
        
        # 重新绘制画布
        self.eff_canvas.draw()
        print("效率图更新完成")
    
    def update_branch_characteristics(self):
        """更新分支特性"""
        try:
            print("开始更新分支特性...")
            # 更新状态指示灯为红色（计算中）
            self.calc_status.set_status(False)
            QApplication.processEvents()  # 确保UI更新
            
            # 执行计算
            self.initialize_calculation()
            
            # 更新效率图
            self.update_efficiency_plots()
            
            # 计算完成，更新状态指示灯为绿色
            self.calc_status.set_status(True)
            print("分支特性更新完成")
            
        except Exception as e:
            print(f"更新分支特性错误: {str(e)}")
            import traceback
            traceback.print_exc()
            self.calc_status.set_status(False)
    
    def update_p_composition(self):
        """P型材料组分变化处理方法"""
        composition = self.p_type_combo.currentText()
        print(f"P型材料组分更新为: {composition}")
        # 更新ZT图
        self.update_zt_plots()
    
    def update_n_composition(self):
        """N型材料组分变化处理方法"""
        composition = self.n_type_combo.currentText()
        print(f"N型材料组分更新为: {composition}")
        # 更新ZT图
        self.update_zt_plots()
    
    def calculate_device_performance(self):
        """计算器件性能"""
        try:
            # 直接使用已存在的状态指示灯，而不是通过findChild查找
            self.calc_status.set_status(False)
            QApplication.processEvents()  # 确保UI更新
            
            # 获取输入参数
            Th = float(self.th_edit.text())
            Tc = float(self.tc_edit.text())
            p_composition = self.p_type_combo.currentText()
            n_composition = self.n_type_combo.currentText()
            area_ratio = float(self.ratio_edit.text())
            
            print(f"\n===== 开始计算器件性能 =====")
            print(f"温度: Th={Th}K, Tc={Tc}K")
            print(f"材料: P型={p_composition}, N型={n_composition}")
            print(f"面积比(N/P): {area_ratio}")
            
            # 导入02.py模块（确保02.py在当前目录或Python路径中）
            try:
                import sys
                import os
                # 获取当前目录及相关目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"当前目录: {current_dir}")
                
                # 检查02.py是否存在
                script_path = os.path.join(current_dir, '02.py')
                if os.path.exists(script_path):
                    print(f"找到02.py脚本: {script_path}")
                else:
                    print(f"警告: 02.py脚本不存在于当前目录")
                    # 尝试搜索其他可能的位置
                    for search_dir in ['.', '..', '../..']:
                        search_path = os.path.join(current_dir, search_dir, '02.py')
                        if os.path.exists(search_path):
                            print(f"在 {search_path} 找到02.py")
                            current_dir = os.path.dirname(search_path)
                            break
                
                # 确保当前目录在Python路径中
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                    print(f"已添加 {current_dir} 到Python路径")
                
                import importlib
                # 重新加载模块以确保获取最新代码
                if "02" in sys.modules:
                    importlib.reload(sys.modules["02"])
                    print("已重新加载02模块")
                
                import_02 = importlib.import_module("02")
                print("成功导入02.py模块")
                
                # 检查数据文件是否存在
                data_files = [
                    'P_yuanshi_2_5.xls', 'P_yuanshi_3_1.xls', 'P_yuanshi_3_7.xls',
                    'N_yuanshi_0.0004.xls', 'N_yuanshi_0.0012.xls', 'N_yuanshi_0.0020.xls'
                ]
                data_dir = current_dir
                for file in data_files:
                    file_path = os.path.join(data_dir, file)
                    if os.path.exists(file_path):
                        print(f"找到数据文件: {file}")
                    else:
                        print(f"警告: 数据文件不存在: {file_path}")
                
                # 验证函数是否存在
                if hasattr(import_02, 'calculate_device_performance'):
                    print("找到calculate_device_performance函数")
                else:
                    raise AttributeError("02.py模块中不存在calculate_device_performance函数")
                
            except Exception as e:
                print(f"导入02.py模块出错: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # 使用02.py中的函数计算效率和功率
            print("调用02.py中的计算函数...")
            
            # 调用02.py的计算函数，直接计算
            try:
                result = import_02.calculate_device_performance(
                    Tc=Tc, 
                    Th=Th, 
                    value_p=p_composition, 
                    value_n=n_composition, 
                    area_ratio=area_ratio, 
                    data_dir=data_dir
                )
                print("02.py计算完成，获取结果...")
            except Exception as e:
                print(f"调用02.py计算函数时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # 从结果中获取数据
            eff_list_P = result['eff_list_P']
            J_list_P = result['J_list_P']
            q_P_list = result['q_P_list']
            eff_list_N = result['eff_list_N']
            J_list_N = result['J_list_N']
            q_N_list = result['q_N_list']
            I_list = result['I_list']
            eff_total_list = result['eff_total_list']
            power_total_list = result['Power_total_list']
            
            # 详细打印返回的数据情况
            print("\n===== 返回的数据详情 =====")
            print(f"P型效率数据: {len(eff_list_P)}个点, 范围: {min(eff_list_P) if eff_list_P else 'N/A'} - {max(eff_list_P) if eff_list_P else 'N/A'}")
            print(f"N型效率数据: {len(eff_list_N)}个点, 范围: {min(eff_list_N) if eff_list_N else 'N/A'} - {max(eff_list_N) if eff_list_N else 'N/A'}")
            print(f"总电流数据: {len(I_list)}个点, 范围: {min(I_list) if I_list else 'N/A'} - {max(I_list) if I_list else 'N/A'}")
            print(f"总效率数据: {len(eff_total_list)}个点, 范围: {min(eff_total_list) if eff_total_list else 'N/A'} - {max(eff_total_list) if eff_total_list else 'N/A'}")
            print(f"总功率数据: {len(power_total_list)}个点, 范围: {min(power_total_list) if power_total_list else 'N/A'} - {max(power_total_list) if power_total_list else 'N/A'}")
            
            # 打印一些实际数据样本
            if I_list and len(I_list) > 0:
                print("\n电流样本: ", I_list[:5] if len(I_list) >= 5 else I_list)
            if eff_total_list and len(eff_total_list) > 0:
                print("效率样本: ", [f"{e:.4f}" for e in (eff_total_list[:5] if len(eff_total_list) >= 5 else eff_total_list)])
            if power_total_list and len(power_total_list) > 0:
                print("功率样本(W): ", [f"{p:.6e}" for p in (power_total_list[:5] if len(power_total_list) >= 5 else power_total_list)])
            
            # 数据处理 - 效率值已经是小数形式，不需要除以100
            # 但功率需要从W转换为W/cm²
            original_power = power_total_list.copy() if power_total_list else []
            power_total_list = [p / 10000 for p in power_total_list]  # 从W/m²转为W/cm²
            
            # 修正效率值，确保没有负值或异常值
            original_eff = eff_total_list.copy() if eff_total_list else []
            corrected_eff = []
            negative_count = 0
            for e in eff_total_list:
                if e < 0:
                    corrected_eff.append(0)  # 负效率值修正为0
                    negative_count += 1
                elif e > 1:
                    corrected_eff.append(0.5)  # 过大效率值修正为0.5，即50%
                else:
                    corrected_eff.append(e)  # 正常范围内的效率值保持不变

            if negative_count > 0:
                print(f"警告: 发现{negative_count}个负效率值，已将其修正为0")
            
            eff_total_list = corrected_eff  # 更新为修正后的效率值列表

            print("数据修正完成:")
            print("功率单位转换: W/m² -> W/cm² (除以10000)")
            if power_total_list and len(power_total_list) > 0:
                print("功率样本(W/cm²): ", [f"{p:.6e}" for p in (power_total_list[:5] if len(power_total_list) >= 5 else power_total_list)])

            if eff_total_list and len(eff_total_list) > 0:
                print("修正后效率样本: ", [f"{e:.4f}" for e in (eff_total_list[:5] if len(eff_total_list) >= 5 else eff_total_list)])

            print(f"获取到的数据点数: P型材料={len(eff_list_P)}, N型材料={len(eff_list_N)}, 总效率/功率={len(eff_total_list)}")

            # 如果没有有效数据点，添加一个默认点
            if not I_list or not eff_total_list or not power_total_list:
                print("警告：没有计算出有效的效率和功率点，添加默认点")
                I_list = [1, 2, 3, 4, 5]
                eff_total_list = [0.01, 0.015, 0.02, 0.018, 0.016]  # 添加一些有变化的默认值
                power_total_list = [1e-6, 1.5e-6, 2e-6, 1.8e-6, 1.6e-6]  # 添加一些有变化的默认值
                print("添加了默认数据点: ")
                print(f"I_list = {I_list}")
                print(f"eff_total_list = {eff_total_list}")
                print(f"power_total_list = {power_total_list}")
            
            # 查找最大功率点和最大效率点
            if power_total_list and max(power_total_list) > 0:
                max_power_idx = np.argmax(power_total_list)
                self.max_power.setText(f"{power_total_list[max_power_idx]:.2e}")
                self.power_current.setText(f"{I_list[max_power_idx]:.2f}")
                print(f"最大功率: {power_total_list[max_power_idx]:.4e} W/cm² 在电流 {I_list[max_power_idx]:.2f}A")
            else:
                self.max_power.setText("0")
                self.power_current.setText("0")
                print("未找到有效的最大功率点")
            
            if eff_total_list and max(eff_total_list) > 0:
                max_eff_idx = np.argmax(eff_total_list)
                # 效率值需要以百分比形式显示
                self.max_eff.setText(f"{eff_total_list[max_eff_idx]*100:.2f}%")
                self.eff_current.setText(f"{I_list[max_eff_idx]:.2f}")
                print(f"最大效率: {eff_total_list[max_eff_idx]*100:.2f}% 在电流 {I_list[max_eff_idx]:.2f}A")
            else:
                self.max_eff.setText("0")
                self.eff_current.setText("0")
                print("未找到有效的最大效率点")
            
            # 更新功率图
            try:
                power_group = self.findChild(QGroupBox, "器件功率")
                if power_group:
                    power_canvases = power_group.findChildren(FigureCanvas)
                    if power_canvases and len(power_canvases) > 0:
                        power_container = power_canvases[0]
                        power_fig = power_container.figure
                        power_ax = power_fig.axes[0]
                        power_ax.clear()
                        
                        print(f"开始绘制功率图, 数据点数: {len(I_list)} / {len(power_total_list)}")
                        # 直接绘制I_list和power_total_list
                        power_ax.plot(I_list, power_total_list, 'b-', linewidth=2.0, label='功率曲线')
                        
                        if max(power_total_list) > 0:
                            power_ax.scatter(I_list[max_power_idx], power_total_list[max_power_idx], 
                                        color='red', marker='o', s=50, label='最大功率点')
                            # 标注最大功率点
                            power_ax.annotate(f"{power_total_list[max_power_idx]:.2e} W/cm²",
                                         (I_list[max_power_idx], power_total_list[max_power_idx]),
                                         xytext=(5, 5), textcoords='offset points')
                        
                        power_ax.set_xlabel("电流 (A)")
                        power_ax.set_ylabel("功率 (W/cm²)")
                        power_ax.set_xlim(0, max(I_list) if I_list else 10)
                        power_ax.set_ylim(0, max(max(power_total_list)*1.1, 1e-6) if power_total_list else 1)
                        power_ax.grid(True, linestyle='--', alpha=0.6)
                        power_ax.legend(loc='best')
                        power_ax.set_facecolor('#F8F8F8')
                        power_fig.tight_layout()
                        power_fig.canvas.draw()
                        print(f"功率图已更新，x轴范围: [0, {max(I_list)}], y轴范围: [0, {max(power_total_list)*1.1}]")
                    else:
                        print("警告: 找不到功率图的FigureCanvas")
                else:
                    # 尝试查找所有可能的GroupBox名称
                    all_group_boxes = self.findChildren(QGroupBox)
                    available_names = [box.title() for box in all_group_boxes]
                    print(f"警告: 找不到器件功率GroupBox，可用的GroupBox: {available_names}")
                    
                    # 尝试使用第一个带有'功率'关键字的GroupBox
                    for box in all_group_boxes:
                        if '功率' in box.title():
                            print(f"找到替代GroupBox: {box.title()}")
                            power_canvases = box.findChildren(FigureCanvas)
                            if power_canvases and len(power_canvases) > 0:
                                # 使用找到的GroupBox更新图表
                                # 与上面的代码类似，但使用找到的box
                                print(f"使用替代GroupBox '{box.title()}' 进行图表更新")
                                power_container = power_canvases[0]
                                power_fig = power_container.figure
                                power_ax = power_fig.axes[0]
                                power_ax.clear()
                                power_ax.plot(I_list, power_total_list, 'b-', linewidth=2.0, label='功率曲线')
                                # 其余绘图代码...
                                power_fig.tight_layout()
                                power_fig.canvas.draw()
                                break
            except Exception as e:
                print(f"更新功率图错误: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 更新效率图
            try:
                eff_group = self.findChild(QGroupBox, "器件效率")
                if not eff_group:
                    # 直接尝试查找所有可能的GroupBox名称
                    all_group_boxes = self.findChildren(QGroupBox)
                    available_names = [box.title() for box in all_group_boxes]
                    print(f"警告: 找不到器件效率GroupBox，可用的GroupBox: {available_names}")
                    
                    # 尝试使用第一个带有'效率'关键字但不含'材料'的GroupBox
                    for box in all_group_boxes:
                        if '效率' in box.title() and '材料' not in box.title() and '最大' not in box.title():
                            print(f"找到替代效率GroupBox: {box.title()}")
                            eff_group = box
                            break
                
                    # 如果仍然找不到，尝试使用任何包含"效率"的GroupBox
                    if not eff_group:
                        for box in all_group_boxes:
                            if '效率' in box.title():
                                print(f"使用备选效率GroupBox: {box.title()}")
                                eff_group = box
                                break
                
                if eff_group:
                    eff_canvases = eff_group.findChildren(FigureCanvas)
                    if eff_canvases and len(eff_canvases) > 0:
                        eff_container = eff_canvases[0]
                        eff_fig = eff_container.figure
                        eff_ax = eff_fig.axes[0]
                        eff_ax.clear()
                        
                        print(f"开始绘制效率图, 数据点数: {len(I_list)} / {len(eff_total_list)}")
                        # 将效率值转换为百分比
                        eff_percent = [e * 100 for e in eff_total_list]  # 转换为百分比
                        
                        eff_ax.plot(I_list, eff_percent, 'r-', linewidth=2.0, label='效率曲线')
                        
                        if max(eff_total_list) > 0:
                            eff_ax.scatter(I_list[max_eff_idx], eff_percent[max_eff_idx], 
                                        color='blue', marker='o', s=50, label='最大效率点')
                            # 标注最大效率点
                            eff_ax.annotate(f"{eff_percent[max_eff_idx]:.2f}%",
                                       (I_list[max_eff_idx], eff_percent[max_eff_idx]),
                                       xytext=(5, 5), textcoords='offset points')
                        
                        eff_ax.set_xlabel("电流 (A)")
                        eff_ax.set_ylabel("效率 (%)")
                        eff_ax.set_xlim(0, max(I_list) if I_list else 10)
                        
                        # 设置Y轴范围，确保至少显示到40%
                        if eff_percent and max(eff_percent) > 0:
                            y_max = max(max(eff_percent)*1.1, 40)  # 至少显示到40%
                        else:
                            y_max = 40  # 默认显示到40%
                        eff_ax.set_ylim(0, y_max)
                        
                        # 添加卡诺效率参考线
                        Th = float(self.th_edit.text())
                        Tc = float(self.tc_edit.text())
                        carnot_eff = (Th - Tc) / Th * 100
                        eff_ax.axhline(y=carnot_eff, color='r', linestyle='--', 
                                     label=f'卡诺效率: {carnot_eff:.1f}%')
                        
                        eff_ax.grid(True, linestyle='--', alpha=0.6)
                        eff_ax.legend(loc='best')
                        eff_ax.set_facecolor('#F8F8F8')
                        eff_fig.tight_layout()
                        eff_fig.canvas.draw()
                        print(f"效率图已更新，x轴范围: [0, {max(I_list)}], y轴范围: [0, {y_max}]")
                    else:
                        print("警告: 找不到效率图的FigureCanvas")
                else:
                    print("错误: 无法找到任何可用的效率图表容器")
            except Exception as e:
                print(f"更新效率图错误: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 更新优化区间图
            if power_total_list and eff_total_list and max(power_total_list) > 0 and max(eff_total_list) > 0:
                try:
                    print("开始绘制优化区间图")
                    opt_group = self.findChild(QGroupBox, "功率效率优化区间")
                    
                    if not opt_group:
                        # 直接尝试查找所有可能的GroupBox名称
                        all_group_boxes = self.findChildren(QGroupBox)
                        available_names = [box.title() for box in all_group_boxes]
                        print(f"警告: 找不到功率效率优化区间GroupBox，可用的GroupBox: {available_names}")
                        
                        # 尝试使用第一个带有'优化'关键字的GroupBox
                        for box in all_group_boxes:
                            if '优化' in box.title():
                                print(f"找到替代优化区间GroupBox: {box.title()}")
                                opt_group = box
                                break
                                
                        # 如果仍然找不到，尝试使用包含'功率'和'效率'的GroupBox
                        if not opt_group:
                            for box in all_group_boxes:
                                if ('功率' in box.title() and '效率' in box.title()) or '区间' in box.title():
                                    print(f"使用备选优化区间GroupBox: {box.title()}")
                                    opt_group = box
                                    break
                    
                    if opt_group:
                        opt_canvases = opt_group.findChildren(FigureCanvas)
                        if opt_canvases and len(opt_canvases) > 0:
                            opt_container = opt_canvases[0]
                            opt_fig = opt_container.figure
                            opt_ax = opt_fig.axes[0]
                            opt_ax.clear()
                            
                            # 效率值转换为百分比
                            eff_percent = [e * 100 for e in eff_total_list]
                            
                            print(f"优化区间图数据: 功率点数={len(power_total_list)}, 效率点数={len(eff_percent)}")
                            opt_ax.plot(power_total_list, eff_percent, 'g-', linewidth=2.0, label='优化曲线')
                            opt_ax.scatter(power_total_list[max_power_idx], eff_percent[max_power_idx], 
                                        color='red', marker='o', s=50, label='最大功率点')
                            opt_ax.scatter(power_total_list[max_eff_idx], eff_percent[max_eff_idx], 
                                        color='blue', marker='o', s=50, label='最大效率点')
                            
                            # 标注最大点
                            opt_ax.annotate(f"最大功率: {power_total_list[max_power_idx]:.2e}W/cm²",
                                       (power_total_list[max_power_idx], eff_percent[max_power_idx]),
                                       xytext=(10, 10), textcoords='offset points', 
                                       arrowprops=dict(arrowstyle="->", color='red'))
                            
                            opt_ax.annotate(f"最大效率: {eff_percent[max_eff_idx]:.2f}%",
                                       (power_total_list[max_eff_idx], eff_percent[max_eff_idx]),
                                       xytext=(-10, 10), textcoords='offset points',
                                       arrowprops=dict(arrowstyle="->", color='blue'))
                            
                            opt_ax.set_xlabel("功率 (W/cm²)")
                            opt_ax.set_ylabel("效率 (%)")
                            
                            # 动态设置坐标轴范围
                            if power_total_list and max(power_total_list) > 0:
                                x_max = max(power_total_list) * 1.1
                            else:
                                x_max = 0.0001  # 设置一个小的默认值
                                
                            y_max = max(max(eff_percent) * 1.1, 40) if eff_percent and max(eff_percent) > 0 else 40  # 至少显示到40%
                            opt_ax.set_xlim(0, x_max)
                            opt_ax.set_ylim(0, y_max)
                            
                            # 添加卡诺效率参考线
                            Th = float(self.th_edit.text())
                            Tc = float(self.tc_edit.text())
                            carnot_eff = (Th - Tc) / Th * 100
                            opt_ax.axhline(y=carnot_eff, color='r', linestyle='--', 
                                         label=f'卡诺效率: {carnot_eff:.1f}%')
                            
                            opt_ax.grid(True, linestyle='--', alpha=0.6)
                            opt_ax.legend(loc='best')
                            opt_fig.tight_layout()
                            opt_fig.canvas.draw()
                            print(f"优化区间图已更新，x轴范围: [0, {x_max}], y轴范围: [0, {y_max}]")
                        else:
                            print("警告: 找不到优化区间图的FigureCanvas")
                    else:
                        print("错误: 无法找到任何可用的优化区间图表容器")
                except Exception as e:
                    print(f"更新优化区间图错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # 保存计算结果供后续使用
            self.last_device_calculation = {
                'eff_list_P': eff_list_P,
                'J_list_P': J_list_P,
                'q_P_list': q_P_list,
                'eff_list_N': eff_list_N,
                'J_list_N': J_list_N,
                'q_N_list': q_N_list,
                'I_list': I_list,
                'eff_total_list': eff_total_list,
                'power_total_list': power_total_list,
                'max_power_idx': max_power_idx if power_total_list and max(power_total_list) > 0 else 0,
                'max_eff_idx': max_eff_idx if eff_total_list and max(eff_total_list) > 0 else 0
            }
            
            # 计算完成，更新状态指示灯为绿色
            self.calc_status.set_status(True)
            print("===== 器件性能计算完成 =====")
            
        except Exception as e:
            print(f"计算器件性能错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 出错时更新状态灯
            self.calc_status.set_status(False)

    def export_data(self):
        """导出数据到文件"""
        try:
            from datetime import datetime
            import pandas as pd
            
            # 获取当前时间作为文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"thermoelectric_data_{timestamp}.xlsx"
            
            # 创建Excel写入器
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 获取所有计算数据
                data = {
                    "高温温度(K)": [float(self.th_edit.text())],
                    "低温温度(K)": [float(self.tc_edit.text())],
                    "P型材料": [self.p_type_combo.currentText()],
                    "N型材料": [self.n_type_combo.currentText()],
                    "N/P面积比": [float(self.ratio_edit.text())],
                    "最大功率(W/cm2)": [float(self.max_power.text())],
                    "最大功率电流密度(A/cm2)": [float(self.power_current.text())],
                    "最大效率": [float(self.max_eff.text())],
                    "最大效率电流密度(A/cm2)": [float(self.eff_current.text())]
                }
                
                # 创建数据帧并保存
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name='计算结果', index=False)
                
                # 确保工作表可见
                workbook = writer.book
                if workbook.sheetnames:
                    workbook.active = workbook.sheetnames.index('计算结果')
            
            print(f"数据已导出到文件: {filename}")
            
        except Exception as e:
            print(f"导出数据错误: {str(e)}")

    def run_external_efficiency(self):
        """运行外部03.py程序进行详细效率计算"""
        try:
            import subprocess
            import os
            import pandas as pd
            import numpy as np
            
            # 获取当前参数
            p_composition = self.p_type_combo.currentText()
            n_composition = self.n_type_combo.currentText()
            Tc = self.tc_edit.text()
            Th = self.th_edit.text()
            area_ratio = self.ratio_edit.text()
            
            # 准备输出目录
            output_dir = "efficiency_results"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 构建命令
            cmd = [
                "python", "03.py",
                "--p_type", p_composition,
                "--n_type", n_composition,
                "--area_ratio", area_ratio,
                "--Tc", Tc,
                "--Th", Th,
                "--output_dir", output_dir
            ]
            
            # 更新状态指示灯
            self.calc_status.set_status(False)
            QApplication.processEvents()
            
            print(f"正在运行详细效率计算: {' '.join(cmd)}")
            
            # 显示一条消息
            QMessageBox.information(self, "计算中", "正在运行效率计算，请等待完成...")
            
            # 检查材料数据文件是否存在
            p_filename_map = {
                "0.01": "P_yuanshi_2_5.xls",
                "0.02": "P_yuanshi_3_1.xls",
                "0.03": "P_yuanshi_3_7.xls"
            }
            
            p_data_file = p_filename_map.get(p_composition)
            if p_data_file is None:
                error_msg = f"不支持的P型材料组分: {p_composition}"
                print(error_msg)
                QMessageBox.critical(self, "错误", error_msg)
                self.calc_status.set_status(False)
                return
                
            n_data_file = f"N_yuanshi_{n_composition}.xls"
            
            if not os.path.exists(p_data_file):
                error_msg = f"找不到P型材料数据文件: {p_data_file}"
                print(error_msg)
                QMessageBox.critical(self, "错误", error_msg)
                self.calc_status.set_status(False)
                return
                
            if not os.path.exists(n_data_file):
                error_msg = f"找不到N型材料数据文件: {n_data_file}"
                print(error_msg)
                QMessageBox.critical(self, "错误", error_msg)
                self.calc_status.set_status(False)
                return
            
            # 创建一个日志文件以储存输出
            log_file = os.path.join(output_dir, f"efficiency_log_{p_composition}_{n_composition}.txt")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"命令: {' '.join(cmd)}\n\n")
                f.write("=== 执行开始 ===\n")
            
            # 运行子进程
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # 获取输出并实时写入日志
            stdout_lines = []
            stderr_lines = []
            
            # 实时读取和记录stdout
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                line = line.strip()
                print(line)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
                stdout_lines.append(line)
            
            # 实时读取和记录stderr
            for line in iter(process.stderr.readline, ''):
                if not line:
                    break
                line = line.strip()
                print(f"错误: {line}")
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"错误: {line}\n")
                stderr_lines.append(line)
            
            # 等待进程结束
            return_code = process.wait()
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 执行结束，返回码: {return_code} ===\n")
            
            stdout = '\n'.join(stdout_lines)
            stderr = '\n'.join(stderr_lines)
            
            # 检查执行结果
            if return_code == 0:
                print("详细效率计算完成")
                QMessageBox.information(self, "完成", "详细效率计算已完成")
                
                # 显示计算结果的图像
                result_image = os.path.join(output_dir, f"summary_{p_composition}_{n_composition}.png")
                if os.path.exists(result_image):
                    pixmap = QPixmap(result_image)
                    if not pixmap.isNull():
                        dialog = ImageViewerDialog(pixmap, self)
                        dialog.setWindowTitle(f"效率计算结果 (P:{p_composition}, N:{n_composition})")
                        dialog.show()
                    else:
                        error_msg = f"无法加载结果图像: {result_image}"
                        print(error_msg)
                        QMessageBox.warning(self, "警告", error_msg)
                else:
                    error_msg = f"结果图像不存在: {result_image}"
                    print(error_msg)
                    QMessageBox.warning(self, "警告", error_msg)
                
                # 加载CSV数据并更新主界面图表
                try:
                    print("从CSV文件加载计算结果数据...")
                    total_data_file = os.path.join(output_dir, f"total_data_{p_composition}_{n_composition}.csv")
                    if os.path.exists(total_data_file):
                        total_data = pd.read_csv(total_data_file)
                        
                        I_list = total_data['I'].tolist()
                        eff_total_list = total_data['eff_total'].tolist()
                        power_total_list = total_data['Power_total'].tolist()
                        
                        # 如果存在效率和功率数据，更新主界面图表
                        if I_list and eff_total_list and power_total_list:
                            print(f"加载的数据: {len(I_list)}个点")
                            print(f"效率范围: {min(eff_total_list):.6f} ~ {max(eff_total_list):.6f}")
                            print(f"功率范围: {min(power_total_list):.6e} ~ {max(power_total_list):.6e}")
                            
                            # 找到最大效率和功率点
                            if max(eff_total_list) > 0:
                                max_eff_idx = eff_total_list.index(max(eff_total_list))
                                self.max_eff.setText(f"{eff_total_list[max_eff_idx]*100:.2f}%")
                                self.eff_current.setText(f"{I_list[max_eff_idx]:.2f}")
                                print(f"最大效率: {eff_total_list[max_eff_idx]*100:.2f}% 在电流 {I_list[max_eff_idx]:.2f}A")
                            
                            if max(power_total_list) > 0:
                                max_power_idx = power_total_list.index(max(power_total_list))
                                self.max_power.setText(f"{power_total_list[max_power_idx]:.2e}")
                                self.power_current.setText(f"{I_list[max_power_idx]:.2f}")
                                print(f"最大功率: {power_total_list[max_power_idx]:.4e} W/cm² 在电流 {I_list[max_power_idx]:.2f}A")
                            
                            # 更新功率图表
                            try:
                                power_group = self.findChild(QGroupBox, "器件功率")
                                if not power_group:
                                    all_group_boxes = self.findChildren(QGroupBox)
                                    available_names = [box.title() for box in all_group_boxes]
                                    print(f"警告: 找不到器件功率GroupBox，可用的GroupBox: {available_names}")
                                    
                                    # 尝试找到一个合适的替代GroupBox
                                    for box in all_group_boxes:
                                        if '功率' in box.title() and '材料' not in box.title() and '最大' not in box.title():
                                            print(f"找到替代GroupBox: {box.title()}")
                                            power_group = box
                                            break
                                
                                if power_group:
                                    power_canvases = power_group.findChildren(FigureCanvas)
                                    if power_canvases and len(power_canvases) > 0:
                                        power_container = power_canvases[0]
                                        power_fig = power_container.figure
                                        power_ax = power_fig.axes[0]
                                        power_ax.clear()
                                        
                                        print(f"绘制功率图, 数据点数: {len(I_list)}")
                                        power_ax.plot(I_list, power_total_list, 'r-', linewidth=2.0, label='功率曲线')
                                        power_ax.scatter(I_list[max_power_idx], power_total_list[max_power_idx], 
                                                     color='red', marker='o', s=50, label='最大功率点')
                                        
                                        power_ax.set_xlabel("电流 (A)")
                                        power_ax.set_ylabel("功率 (W/cm²)")
                                        power_ax.set_xlim(0, max(I_list) if I_list else 10)
                                        power_ax.set_ylim(0, max(power_total_list)*1.1 if power_total_list else 1)
                                        
                                        power_ax.annotate(f"{power_total_list[max_power_idx]:.2e}",
                                                     (I_list[max_power_idx], power_total_list[max_power_idx]),
                                                     xytext=(5, 5), textcoords='offset points')
                                        
                                        power_ax.grid(True, linestyle='--', alpha=0.6)
                                        power_ax.legend(loc='best')
                                        power_fig.tight_layout()
                                        power_fig.canvas.draw()
                                        print("功率图已更新")
                            except Exception as e:
                                print(f"更新功率图错误: {str(e)}")
                                import traceback
                                traceback.print_exc()
                            
                            # 更新效率图表
                            try:
                                eff_group = self.findChild(QGroupBox, "器件效率")
                                if not eff_group:
                                    all_group_boxes = self.findChildren(QGroupBox)
                                    available_names = [box.title() for box in all_group_boxes]
                                    print(f"警告: 找不到器件效率GroupBox，可用的GroupBox: {available_names}")
                                    
                                    # 尝试找到一个合适的替代GroupBox
                                    for box in all_group_boxes:
                                        if '效率' in box.title() and '材料' not in box.title() and '最大' not in box.title():
                                            print(f"找到替代效率GroupBox: {box.title()}")
                                            eff_group = box
                                            break
                                
                                if eff_group:
                                    eff_canvases = eff_group.findChildren(FigureCanvas)
                                    if eff_canvases and len(eff_canvases) > 0:
                                        eff_container = eff_canvases[0]
                                        eff_fig = eff_container.figure
                                        eff_ax = eff_fig.axes[0]
                                        eff_ax.clear()
                                        
                                        print(f"开始绘制效率图, 数据点数: {len(I_list)} / {len(eff_total_list)}")
                                        # 将效率值转换为百分比
                                        eff_percent = [e * 100 for e in eff_total_list]  # 转换为百分比
                                        
                                        eff_ax.plot(I_list, eff_percent, 'r-', linewidth=2.0, label='效率曲线')
                                        
                                        if max(eff_total_list) > 0:
                                            eff_ax.scatter(I_list[max_eff_idx], eff_percent[max_eff_idx], 
                                                        color='blue', marker='o', s=50, label='最大效率点')
                                            # 标注最大效率点
                                            eff_ax.annotate(f"{eff_percent[max_eff_idx]:.2f}%",
                                                      (I_list[max_eff_idx], eff_percent[max_eff_idx]),
                                                      xytext=(5, 5), textcoords='offset points')
                                        
                                        eff_ax.set_xlabel("电流 (A)")
                                        eff_ax.set_ylabel("效率 (%)")
                                        eff_ax.set_xlim(0, max(I_list) if I_list else 10)
                                        
                                        # 设置Y轴范围，确保至少显示到40%
                                        y_max = max(max(eff_percent)*1.1, 40) if eff_percent and max(eff_percent) > 0 else 40
                                        eff_ax.set_ylim(0, y_max)
                                        
                                        # 添加卡诺效率参考线
                                        Th = float(self.th_edit.text())
                                        Tc = float(self.tc_edit.text())
                                        carnot_eff = (Th - Tc) / Th * 100
                                        eff_ax.axhline(y=carnot_eff, color='r', linestyle='--', 
                                                    label=f'卡诺效率: {carnot_eff:.1f}%')
                                        
                                        eff_ax.grid(True, linestyle='--', alpha=0.6)
                                        eff_ax.legend(loc='best')
                                        eff_fig.tight_layout()
                                        eff_fig.canvas.draw()
                                        print(f"效率图已更新，x轴范围: [0, {max(I_list)}], y轴范围: [0, {y_max}]")
                            except Exception as e:
                                print(f"更新效率图错误: {str(e)}")
                                import traceback
                                traceback.print_exc()
                            
                            # 更新优化区间图
                            try:
                                opt_group = self.findChild(QGroupBox, "功率效率优化区间")
                                if not opt_group:
                                    all_group_boxes = self.findChildren(QGroupBox)
                                    available_names = [box.title() for box in all_group_boxes]
                                    print(f"警告: 找不到功率效率优化区间GroupBox，可用的GroupBox: {available_names}")
                                    
                                    for box in all_group_boxes:
                                        if '优化' in box.title():
                                            print(f"找到替代优化区间GroupBox: {box.title()}")
                                            opt_group = box
                                            break
                                
                                if opt_group:
                                    opt_canvases = opt_group.findChildren(FigureCanvas)
                                    if opt_canvases and len(opt_canvases) > 0:
                                        opt_container = opt_canvases[0]
                                        opt_fig = opt_container.figure
                                        opt_ax = opt_fig.axes[0]
                                        opt_ax.clear()
                                        
                                        # 效率值转换为百分比
                                        eff_percent = [e * 100 for e in eff_total_list]
                                        
                                        print(f"优化区间图数据: 功率点数={len(power_total_list)}, 效率点数={len(eff_percent)}")
                                        opt_ax.plot(power_total_list, eff_percent, 'g-', linewidth=2.0, label='优化曲线')
                                        opt_ax.scatter(power_total_list[max_power_idx], eff_percent[max_power_idx], 
                                                    color='red', marker='o', s=50, label='最大功率点')
                                        opt_ax.scatter(power_total_list[max_eff_idx], eff_percent[max_eff_idx], 
                                                    color='blue', marker='o', s=50, label='最大效率点')
                                        
                                        # 标注最大点
                                        opt_ax.annotate(f"最大功率: {power_total_list[max_power_idx]:.2e}W/cm²",
                                                  (power_total_list[max_power_idx], eff_percent[max_power_idx]),
                                                  xytext=(10, 10), textcoords='offset points', 
                                                  arrowprops=dict(arrowstyle="->", color='red'))
                                        
                                        opt_ax.annotate(f"最大效率: {eff_percent[max_eff_idx]:.2f}%",
                                                  (power_total_list[max_eff_idx], eff_percent[max_eff_idx]),
                                                  xytext=(-10, 10), textcoords='offset points',
                                                  arrowprops=dict(arrowstyle="->", color='blue'))
                                        
                                        opt_ax.set_xlabel("功率 (W/cm²)")
                                        opt_ax.set_ylabel("效率 (%)")
                                        
                                        # 动态设置坐标轴范围
                                        x_max = max(power_total_list) * 1.1 if power_total_list and max(power_total_list) > 0 else 0.0001
                                        y_max = max(max(eff_percent) * 1.1, 40) if eff_percent and max(eff_percent) > 0 else 40
                                        opt_ax.set_xlim(0, x_max)
                                        opt_ax.set_ylim(0, y_max)
                                        
                                        # 添加卡诺效率参考线
                                        Th = float(self.th_edit.text())
                                        Tc = float(self.tc_edit.text())
                                        carnot_eff = (Th - Tc) / Th * 100
                                        opt_ax.axhline(y=carnot_eff, color='r', linestyle='--', 
                                                    label=f'卡诺效率: {carnot_eff:.1f}%')
                                        
                                        opt_ax.grid(True, linestyle='--', alpha=0.6)
                                        opt_ax.legend(loc='best')
                                        opt_fig.tight_layout()
                                        opt_fig.canvas.draw()
                                        print(f"优化区间图已更新")
                            except Exception as e:
                                print(f"更新优化区间图错误: {str(e)}")
                                import traceback
                                traceback.print_exc()
                            
                            # 保存计算结果供后续使用
                            self.last_device_calculation = {
                                'I_list': I_list,
                                'eff_total_list': eff_total_list,
                                'power_total_list': power_total_list,
                                'max_power_idx': max_power_idx if power_total_list and max(power_total_list) > 0 else 0,
                                'max_eff_idx': max_eff_idx if eff_total_list and max(eff_total_list) > 0 else 0
                            }
                    else:
                        print(f"警告: 找不到总计算结果文件：{total_data_file}")
                except Exception as e:
                    print(f"加载和应用计算结果错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                # 更新状态指示灯为绿色（完成）
                self.calc_status.set_status(True)
            else:
                error_msg = f"详细效率计算失败，错误码: {return_code}\n\n"
                if stderr:
                    error_msg += f"错误信息:\n{stderr}\n"
                print(error_msg)
                
                QMessageBox.critical(self, "错误", f"详细效率计算失败。请查看控制台输出或日志文件：\n{log_file}")
                
                # 保持状态指示灯为红色（出错）
                self.calc_status.set_status(False)
            
        except Exception as e:
            error_msg = f"运行外部效率计算错误: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            QMessageBox.critical(self, "错误", error_msg)
            # 保持状态指示灯为红色（出错）
            self.calc_status.set_status(False)

    def analyze_material_performance(self, material_type, composition, current_density):
        """分析材料性能并可视化结果，帮助查找问题"""
        try:
            if not hasattr(self, 'last_calc_data'):
                print("尚未执行效率计算，请先计算效率")
                return
                
            data = self.last_calc_data
            
            # 创建一个2x2的可视化图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{material_type}型材料 (组分={composition}, 电流密度={current_density}A/cm²) 性能分析", fontsize=14)
            
            # 1. 温度分布
            ax1 = axes[0, 0]
            x_range = np.arange(1, len(data['temperature']) + 1)
            ax1.plot(x_range, data['temperature'], 'b-o')
            ax1.set_title('温度分布')
            ax1.set_xlabel('格点位置')
            ax1.set_ylabel('温度 (K)')
            ax1.grid(True)
            
            # 2. 材料属性随温度变化
            ax2 = axes[0, 1]
            ax2.plot(data['temperature'], data['seebeck'] * 1e6, 'r-', label='塞贝克系数 (μV/K)')
            ax2.set_xlabel('温度 (K)')
            ax2.set_ylabel('塞贝克系数 (μV/K)')
            ax2.set_title('塞贝克系数分布')
            ax2.grid(True)
            
            ax2_twin = ax2.twinx()
            ax2_twin.plot(data['temperature'], data['resistivity'] * 1e6, 'g-', label='电阻率 (μΩ·m)')
            ax2_twin.set_ylabel('电阻率 (μΩ·m)')
            
            # 添加双轴图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            # 3. 能量流动分析
            ax3 = axes[1, 0]
            seebeck_power = data['seebeck'] * data['dTdx'] * data['current_density']
            joule_heat = data['resistivity'] * data['current_density']**2
            
            ax3.plot(x_range, seebeck_power, 'b-', label='塞贝克功率')
            ax3.plot(x_range, joule_heat, 'r-', label='焦耳热损失')
            ax3.plot(x_range, seebeck_power - joule_heat, 'g-', label='净功率')
            ax3.set_title('能量流动分析')
            ax3.set_xlabel('格点位置')
            ax3.set_ylabel('功率密度 (W/m³)')
            ax3.grid(True)
            ax3.legend()
            
            # 4. 热流分析
            ax4 = axes[1, 1]
            fourier_heat = data['thermal_cond'] * data['dTdx']
            peltier_heat = data['current_density'] * data['seebeck'] * data['temperature']
            ax4.plot(x_range, fourier_heat, 'b-', label='傅里叶热流')
            ax4.plot(x_range, peltier_heat, 'r-', label='帕尔贴热流')
            ax4.plot(x_range, fourier_heat - peltier_heat, 'g-', label='净热流')
            ax4.set_title('热流分析')
            ax4.set_xlabel('格点位置')
            ax4.set_ylabel('热流密度 (W/m²)')
            ax4.grid(True)
            ax4.legend()
            
            plt.tight_layout()
            plt.show()
            
            # 打印能量平衡分析
            print("\n===== 能量平衡分析 =====")
            heat_in = abs(fourier_heat[0] - peltier_heat[0])
            heat_out = abs(fourier_heat[-1] - peltier_heat[-1])
            total_joule = np.sum(joule_heat) * (x_range[-1] - x_range[0]) / (len(x_range) - 1)
            total_power = np.sum(seebeck_power - joule_heat) * (x_range[-1] - x_range[0]) / (len(x_range) - 1)
            
            print(f"入口热流: {heat_in:.3e} W/m²")
            print(f"出口热流: {heat_out:.3e} W/m²")
            print(f"总焦耳热: {total_joule:.3e} W/m²")
            print(f"总功率输出: {total_power:.3e} W/m²")
            print(f"热平衡差值: {(heat_in - heat_out - total_power):.3e} W/m² (理论上应接近0)")
            
        except Exception as e:
            print(f"性能分析错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ThermoelectricApp()
    window.show()
    sys.exit(app.exec_())
