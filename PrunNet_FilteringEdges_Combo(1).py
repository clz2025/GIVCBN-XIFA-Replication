import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm


def PrunNet_FilteringEdges(input_file_path, output_folder_path):
    """
    主函数：兼顾行和列的边过滤网络（带输入输出功能）

    参数:
        input_file_path - 输入文件路径
        output_folder_path - 输出文件夹路径

    返回:
        FE - 过滤后的网络（带节点名称的DataFrame）
    """

    print("开始读取输入文件...")
    start_time = time.time()

    # 读取输入文件
    try:
        # 尝试读取CSV文件
        if input_file_path.endswith('.csv'):
            data = pd.read_csv(input_file_path, index_col=0)
        # 尝试读取Excel文件
        elif input_file_path.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(input_file_path, index_col=0)
        else:
            # 默认尝试读取CSV
            data = pd.read_csv(input_file_path, index_col=0)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    # 转换为numpy数组
    net = data.values.astype(float)
    print(f"数据读取完成，矩阵形状: {net.shape}")

    read_time = time.time() - start_time
    print(f"文件读取时间: {read_time:.2f} 秒")

    # 对行进行筛选
    print("开始行筛选...")
    PrunNet1, _, _ = PrunNet_from_XI(net)

    # 对列进行筛选（通过转置）
    print("开始列筛选...")
    PrunNet2, _, _ = PrunNet_from_XI(net.T)

    # 二者合并
    overlap = PrunNet1 + PrunNet2.T

    # 变为布尔网络
    bnet = (overlap != 0).astype(float)

    # 计算最终结果
    FE_matrix = bnet * net

    # 创建结果DataFrame，保留原始行列名
    FE = pd.DataFrame(FE_matrix, index=data.index, columns=data.columns)

    # 生成输出文件名：输入文件名 + FE
    input_filename = os.path.basename(input_file_path)  # 获取文件名（含扩展名）
    input_name = os.path.splitext(input_filename)[0]    # 去掉扩展名
    output_filename = f"{input_name}-FE.csv"
    output_file_path = os.path.join(output_folder_path, output_filename)

    # 保存结果为CSV格式
    print("开始保存结果...")
    save_start_time = time.time()

    # 确保输出文件夹存在
    os.makedirs(output_folder_path, exist_ok=True)

    # 保存为CSV文件
    FE.to_csv(output_file_path)

    save_time = time.time() - save_start_time
    total_time = time.time() - start_time

    # 计算统计信息（仅用于显示，不保存到文件）
    NeFE = np.sum(bnet)
    WratioFE = 100 * np.sum(FE_matrix) / np.sum(net)

    print(f"结果保存完成: {output_file_path}")
    print(f"文件保存时间: {save_time:.2f} 秒")
    print(f"总计算时间: {total_time:.2f} 秒")
    print(f"原始网络边数: {np.sum(net != 0)}")
    print(f"过滤后边数: {NeFE}")
    print(f"权重保留比例: {WratioFE:.2f}%")

    return FE


def PrunNet_from_XI(net):
    """
    子函数：基于X指数进行网络修剪

    参数:
        net - 网络矩阵 (numpy数组)

    返回:
        PrunNet - 修剪后的网络
        Ne - 修剪后边的数量
        Wratio - 权重保留比例
    """

    M, N = net.shape
    _, A = X_Index(net)
    PrunNet = np.zeros((M, N))  # 预分配矩阵

    # 使用tqdm显示进度条
    for i in tqdm(range(M), desc="处理行"):
        # 对第i行进行降序排序，返回索引
        Xidx = np.argsort(net[i, :])[::-1]
        # 创建反向索引映射
        Xidx2 = np.argsort(Xidx)

        for j in range(N):
            if Xidx2[j] < A[i]:  # Python索引从0开始，所以用<而不是<=
                PrunNet[i, j] = net[i, j]
            else:
                PrunNet[i, j] = 0

    bnet = (PrunNet != 0).astype(float)
    Ne = np.sum(bnet)
    Wratio = 100 * np.sum(PrunNet) / np.sum(net)

    return PrunNet, Ne, Wratio


def X_Index(net):
    """
    子函数：计算X指数

    参数:
        net - 网络矩阵 (numpy数组)

    返回:
        XI - X指数值
        A - 阈值索引
    """

    M, N = net.shape
    XI = np.zeros(M)
    A = np.zeros(M, dtype=int)  # 确保A是整数类型

    # 使用tqdm显示进度条
    for i in tqdm(range(M), desc="计算X指数"):
        # 对第i行进行降序排序
        VEC = np.sort(net[i, :])[::-1]
        W = np.sum(VEC)

        # 处理全零行的情况
        if np.sum(net[i, :]) == 0:
            XI[i] = 0
            A[i] = 0
        else:
            for a in range(1, N + 1):  # a从1到N
                if (np.sum(VEC[:a]) / W >= 1 - a / N and
                        (a == 1 or np.sum(VEC[:a - 1]) / W < 1 - (a - 1) / N)):
                    XI[i] = 100 * a / N
                    A[i] = a
                    break  # 找到符合条件的a后就跳出循环

    return XI, A


if __name__ == '__main__':
    """
    注意：
    1. 输入文件格式支持.csv, .xlsx, .xls，第一行和第一列为节点名称
    2. 输出文件格式为.csv，只包含过滤后的网络FE
    3. 确保输出文件夹路径存在
    """

    # 设置输入文件路径和输出文件夹路径
    input_file = r"C:\Users\lenovo\Desktop\基于GIVCN-ADB-FE模型的NODF计算结果\新建文件夹\2024_Scenario3_DropRatio0.2.csv"
    output_folder = r"C:\Users\lenovo\Desktop\基于GIVCN-ADB-FE模型的NODF计算结果\新建文件夹"

    # 调用主函数
    FE = PrunNet_FilteringEdges(input_file, output_folder)

    if FE is not None:
        print("\n计算完成！")
        # 输出文件路径会在函数内部生成并显示
    else:
        print("计算过程中出现错误！")