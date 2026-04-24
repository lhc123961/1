import cv2
import numpy as np
import os

def calculate_fft_frequency(block):
    """方法一：FFT 方法，找到包含 95% 能量的最高频率"""
    # 1. 做 FFT
    f = np.fft.fft2(block)
    fshift = np.fft.fftshift(f)  # 将低频移到中心

    # 2. 计算功率谱 (Power Spectrum)
    power_spectrum = np.abs(fshift) ** 2

    # 3. 创建频率坐标网格
    rows, cols = block.shape
    crow, ccol = rows // 2, cols // 2  # 中心点

    # 计算每个点到中心的距离（即频率半径）
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)

    # 4. 按频率距离排序，累加能量
    dist_flat = dist_from_center.flatten()
    ps_flat = power_spectrum.flatten()

    # 按距离（频率）从小到大排序
    sorted_indices = np.argsort(dist_flat)
    sorted_dist = dist_flat[sorted_indices]
    sorted_ps = ps_flat[sorted_indices]

    # 计算总能量
    total_energy = np.sum(sorted_ps)
    target_energy = 0.95 * total_energy

    # 累加找到 95% 的位置
    current_energy = 0
    cutoff_radius = 0
    for i in range(len(sorted_ps)):
        current_energy += sorted_ps[i]
        if current_energy >= target_energy:
            cutoff_radius = sorted_dist[i]
            break

    return cutoff_radius


def calculate_gradient_frequency(block):
    """方法二：梯度方法，计算均方根频率"""
    # 1. 计算梯度 (Sobel算子)
    grad_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)

    # 2. 计算梯度的平方模长 |grad I|^2
    grad_squared = grad_x**2 + grad_y**2

    # 3. 计算期望 E[|grad I|^2] (即平均值)
    e_grad_squared = np.mean(grad_squared)

    # 4. 计算方差 Var(I)
    var_i = np.var(block)

    # 防止除以 0 (如果是纯色块)
    if var_i < 1e-5:
        return 0

    # 5. 代入公式
    numerator = e_grad_squared
    denominator = (2 * np.pi)**2 * var_i

    f_rms = np.sqrt(numerator / denominator)

    return f_rms


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 指定图片的绝对路径
    
    img_path = r"C:\war3\test.jpg"

    # 2. 读取图片
    img = cv2.imread(img_path)

    # 3. 检查图片是否加载成功
    if img is None:
        print(f"错误：无法读取图片，请检查路径是否正确：{img_path}")
    else:
        print(f"✅ 图片读取成功！图片尺寸：{img.shape}")

        # 4. 转为灰度图（因为频率计算通常基于单通道）
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # 5. 设置分块大小 (例如 64x64)
        block_size = 64
        h, w = img_gray.shape

        print(f"开始计算频率... (分块大小: {block_size}x{block_size})")

        # 6. 遍历图片进行分块计算
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img_gray[y:y+block_size, x:x+block_size]

                # 计算 FFT 频率
                fft_freq = calculate_fft_frequency(block)

                # 计算梯度频率
                grad_freq = calculate_gradient_frequency(block)

                print(f"位置 ({y}, {x}): FFT频率半径={fft_freq:.2f}, 梯度均方根频率={grad_freq:.2f}")

                # 如果只想看第一块的结果，可以在这里 break
                # break