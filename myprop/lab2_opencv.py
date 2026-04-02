import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1. 自定义实现模块：手动直方图均衡化
# ==========================================
def manual_histogram_equalization(img):
    """
    使用 NumPy 手动实现全局直方图均衡化
    """
    # 计算直方图 (256个bins)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # 计算累积分布函数 (CDF)
    cdf = hist.cumsum()

    # 掩膜处理：忽略直方图为0的部分
    cdf_m = np.ma.masked_equal(cdf, 0)

    # 均衡化公式: cdf(x) * (L-1) / (M*N)
    # 归一化到 0-255
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # 填充回原始数据类型
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # 映射原图像素到新值
    img_eq = cdf[img]

    return img_eq

# ==========================================
# 2. 评价指标计算
# ==========================================
def calculate_metrics(img):
    """
    计算清晰度(拉普拉斯方差) 和 信息熵
    """
    # 1. 清晰度评价 (Laplacian Variance) - 值越大越清晰
    clarity = cv2.Laplacian(img, cv2.CV_64F).var()

    # 2. 信息熵 (Entropy) - 值越大信息量越丰富
    # 计算直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum() # 概率分布
    # 过滤掉概率为0的项以避免log(0)
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))

    return clarity, entropy

# ==========================================
# 3. 主处理流程
# ==========================================
def process_image(image_path):
    print(f"--- 正在处理: {image_path} ---")

    # 读取图像 (灰度模式)
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    # --- 方法实现 ---

    # 1. 全局直方图均衡化 (调用自定义函数)
    img_global_eq = manual_histogram_equalization(img)

    # 2. CLAHE (自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # 3. 均值滤波
    img_mean = cv2.blur(img, (5, 5))

    # 4. 高斯滤波
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # 5. 中值滤波
    img_median = cv2.medianBlur(img, 5)

    # 6. 锐化方法 (使用拉普拉斯算子增强)
    # 原理: Original + Amount * SharpeningMask
    gaussian_blur = cv2.GaussianBlur(img, (9, 9), 10.0)
    img_sharpen = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)

    # 7. 组合处理: 滤波 -> 均衡 (先中值去噪，再CLAHE增强对比度)
    img_combo_1 = clahe.apply(img_median)

    # 8. 组合处理: 均衡 -> 滤波 (先CLAHE增强，再中值去噪)
    img_combo_2 = cv2.medianBlur(img_clahe, 5)

    # --- 计算评价指标 ---
    methods = {
        "Original": img,
        "Manual Global Eq": img_global_eq,
        "CLAHE": img_clahe,
        "Mean Filter": img_mean,
        "Gaussian Filter": img_gaussian,
        "Median Filter": img_median,
        "Sharpening": img_sharpen,
        "Filter->Equal (Combo1)": img_combo_1,
        "Equal->Filter (Combo2)": img_combo_2
    }

    # --- 可视化展示 ---
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Results for {image_path}', fontsize=16)

    for i, (name, result_img) in enumerate(methods.items()):
        clarity, entropy = calculate_metrics(result_img)

        plt.subplot(3, 3, i + 1)
        plt.imshow(result_img, cmap='gray')
        plt.title(f"{name}\nClarity: {clarity:.1f}, Entropy: {entropy:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 运行程序
# ==========================================
if __name__ == "__main__":
    # 请在这里填入你的图片路径
    # 为了演示，这里假设你有三张图片
    image_list = ['test1.jpg', 'test2.jpg', 'test3.jpg']

    # 检查文件是否存在，如果不存在则生成一张模拟的低对比度图片用于演示
    import os
    valid_images = []
    for img_name in image_list:
        if os.path.exists(img_name):
            valid_images.append(img_name)

    if not valid_images:
        print("未找到测试图片，正在生成一张模拟低对比度图片用于演示...")
        dummy_img = np.random.randint(50, 100, (300, 300), dtype=np.uint8) # 低对比度噪声图
        cv2.imwrite('dummy_test.jpg', dummy_img)
        process_image('dummy_test.jpg')
    else:
        for img_path in valid_images:
            process_image(img_path)