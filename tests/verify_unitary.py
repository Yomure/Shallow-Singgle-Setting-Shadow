import numpy as np

def generate_sic_matrix(c):
    """
    根据给定的参数 c 生成 SIC 矩阵。
    """
    # 定义基础常数
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    
    # 定义指数中的相位部分
    # 注意 (-1)^c
    phase_sign = (-1)**c
    theta_1 = phase_sign * (2 * np.pi / 3)
    theta_2 = phase_sign * (np.pi / 3)
    
    # 构造矩阵元素 (忽略外部的 1/sqrt2，最后再乘)
    
    # 第一行: [1, 0, 1, 0]
    row1 = [1, 0, 1, 0]
    
    # 第二行: [1/sqrt3, sqrt2/sqrt3, -1/sqrt3, sqrt2/sqrt3]
    row2 = [1/sqrt3, sqrt2/sqrt3, -1/sqrt3, sqrt2/sqrt3]
    
    # 第三行
    # col2: e^{i * theta_1} * sqrt2/sqrt3
    # col4: -e^{i * theta_2} * sqrt2/sqrt3
    r3_c2 = np.exp(1j * theta_1) * (sqrt2 / sqrt3)
    r3_c4 = -np.exp(1j * theta_2) * (sqrt2 / sqrt3)
    row3 = [1/sqrt3, r3_c2, -1/sqrt3, r3_c4]
    
    # 第四行 (指数带负号)
    # col2: e^{-i * theta_1} * sqrt2/sqrt3
    # col4: -e^{-i * theta_2} * sqrt2/sqrt3
    r4_c2 = np.exp(-1j * theta_1) * (sqrt2 / sqrt3)
    r4_c4 = -np.exp(-1j * theta_2) * (sqrt2 / sqrt3)
    row4 = [1/sqrt3, r4_c2, -1/sqrt3, r4_c4]
    
    # 组合成矩阵并乘以系数 1/sqrt2
    matrix = np.array([row1, row2, row3, row4]) * (1 / sqrt2)
    
    return matrix

def verify_unitary(matrix):
    """
    验证矩阵是否为酉矩阵: U^dagger @ U == I
    """
    # 计算共轭转置 (Conjugate Transpose)
    matrix_dagger = matrix.conj().T
    
    # 矩阵乘法
    product = np.dot(matrix_dagger, matrix)
    
    # 创建 4x4 单位矩阵
    identity = np.eye(4)
    
    # 检查是否近似相等 (由于浮点数精度问题，不能用 ==)
    is_unitary = np.allclose(product, identity, atol=1e-10)
    
    return is_unitary, product

# --- 测试 ---

# 假设 c = 0 (通常 c 是整数)
c_value = 1
U = generate_sic_matrix(c_value)

is_unitary, result_product = verify_unitary(U)

print(f"当 c={c_value} 时:")
print("矩阵是否为酉矩阵 (Unitary)? ->", "【是】" if is_unitary else "【否】")

print("\nU^dagger @ U 的计算结果 (保留3位小数):")
#为了显示整洁，我们将极小的虚部和误差截断显示
print(np.round(result_product.real, 3) + 1j * np.round(result_product.imag, 3))

# 测试另一个 c 值
c_value = 1
U_2 = generate_sic_matrix(c_value)
is_unitary_2, _ = verify_unitary(U_2)
print(f"\n当 c={c_value} 时:")
print("矩阵是否为酉矩阵? ->", "【是】" if is_unitary_2 else "【否】")