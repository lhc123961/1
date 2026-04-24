import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

a4_image_path = r"C:\Users\刘浩成\source\myprop\main.cpp\计算机视觉实验作业5.刘浩成2023100475\A4.jpg"

# 检查文件是否存在
if not os.path.exists(a4_image_path):
    print(f"警告：找不到图片文件 -> {a4_image_path}")
    print("请确保路径正确，或者将 A4.jpg 放在与脚本相同的文件夹下。")
else:
    print(f" 成功找到图片：{a4_image_path}")

#
def create_test_image():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    # 画矩形(红)、圆(蓝)、平行线(绿)、垂直线(黑十字)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 3)
    cv2.circle(img, (200, 200), 50, (255, 0, 0), 3)
    cv2.line(img, (50, 50), (350, 50), (0, 255, 0), 2)
    cv2.line(img, (50, 80), (350, 80), (0, 255, 0), 2)
    cv2.line(img, (150, 320), (150, 380), (0, 0, 0), 2)
    cv2.line(img, (120, 350), (180, 350), (0, 0, 0), 2)
    return img

def run_geometry_demo():
    original_img = create_test_image()
    rows, cols = original_img.shape[:2]

    # 1. 相似变换 (旋转30度 + 缩放0.8)
    M_sim = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 0.8)
    img_sim = cv2.warpAffine(original_img, M_sim, (cols, rows))

    # 2. 仿射变换 (倾斜)
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    pts2 = np.float32([[10,100], [200,50], [100,250]])
    M_aff = cv2.getAffineTransform(pts1, pts2)
    img_aff = cv2.warpAffine(original_img, M_aff, (cols, rows))

    # 3. 透视变换
    pts1 = np.float32([[50,50], [350,50], [50,350], [350,350]])
    pts2 = np.float32([[0,0], [300,0], [0,300], [250,280]])
    M_per = cv2.getPerspectiveTransform(pts1, pts2)
    img_per = cv2.warpPerspective(original_img, M_per, (cols, rows))

    # 显示结果
    images = [original_img, img_sim, img_aff, img_per]
    titles = ['原图', '相似变换', '仿射变换', '透视变换']

    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        # OpenCV是BGR，Matplotlib是RGB，需要转换
        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 第二部分：A4 纸图像校正实验 
# ---------------------------------------------------------
def run_a4_correction():
    if not os.path.exists(a4_image_path):
        return

    # 1. 读取 A4 图片
    img = cv2.imdecode(np.fromfile(a4_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("无法读取图片，请检查格式。")
        return

    # 获取图片尺寸
    h, w = img.shape[:2]

    # 2. 模拟“透视畸变” (
    # 定义原图的四个角
    pts_original = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # 定义畸变后的四个角 (模拟近大远小)
    pts_distorted = np.float32([
        [w*0.1, h*0.1],       # 左上
        [w*0.9, h*0.15],      # 右上
        [w*0.15, h*0.9],      # 左下
        [w*0.85, h*0.85]      # 右下
    ])

    # 计算畸变矩阵并应用
    M_distort = cv2.getPerspectiveTransform(pts_original, pts_distorted)
    img_distorted = cv2.warpPerspective(img, M_distort, (w, h))

    # 3. 图像校正 
    M_correct = cv2.getPerspectiveTransform(pts_distorted, pts_original)
    img_corrected = cv2.warpPerspective(img_distorted, M_correct, (w, h))

    # 4. 显示校正前后的对比
    # 将 BGR 转为 RGB 用于显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_distorted_rgb = cv2.cvtColor(img_distorted, cv2.COLOR_BGR2RGB)
    img_corrected_rgb = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("原始 A4 图")
    plt.axis('off')



    plt.subplot(1, 3, 3)
    plt.imshow(img_corrected_rgb)
    plt.title("校正后的结果")
    plt.axis('off')

    plt.suptitle("A4 纸透视畸变与校正实验")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 主程序运行
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- 正在运行几何变换演示 ---")
    run_geometry_demo()

    print("\n--- 正在运行 A4 纸校正实验 ---")
    run_a4_correction()