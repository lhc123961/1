#include <iostream>
#include <opencv2/opencv.hpp> // 引入 OpenCV 头文件

using namespace std;
using namespace cv;

int main() {
    // --- 任务 1: 使用 OpenCV 读取一张测试图片 ---
    // 请确保你的项目目录下有一张名为 "test.jpg" 的图片，或者修改这里的路径
    Mat img = imread("test.jpg");

    // 检查图片是否读取成功
    if (img.empty()) {
        cout << "错误：无法读取图片，请检查文件名或路径！" << endl;
        return -1;
    }
    cout << ">>> 图片读取成功！" << endl;

    // --- 任务 2: 输出图像基本信息 ---
    cout << "--- 图像基本信息 ---" << endl;
    // 图像尺寸 (列数=宽度, 行数=高度)
    cout << "图像尺寸 (宽 x 高): " << img.cols << " x " << img.rows << endl;
    // 图像通道数
    cout << "图像通道数: " << img.channels() << endl;
    // 图像数据类型 (例如 CV_8UC3)
    cout << "图像数据类型: " << img.type() << endl;

    // --- 任务 3: 显示原图 ---
    // 创建一个窗口
    namedWindow("原图", WINDOW_AUTOSIZE);
    // 在窗口中显示图像
    imshow("原图", img);

    // --- 任务 4: 转换为灰度图 ---
    Mat gray_img;
    // 将 BGR 彩色图转换为灰度图
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    cout << "--- 转换后灰度图信息 ---" << endl;
    cout << "灰度图通道数: " << gray_img.channels() << endl;

    // 显示灰度图
    namedWindow("灰度图", WINDOW_AUTOSIZE);
    imshow("灰度图", gray_img);

    // --- 任务 5: 保存处理结果 ---
    // 将灰度图保存为新文件
    bool is_saved = imwrite("result_gray.jpg", gray_img);
    if (is_saved) {
        cout << ">>> 灰度图已保存为 result_gray.jpg" << endl;
    }

    // --- 任务 6: 用 NumPy 做一个简单操作 (C++ 中对应 Mat 的 ROI 操作) ---
    // 题目要求：把图像左上角一块区域裁剪出来保存
    // 假设裁剪左上角 100x100 的区域 (注意：需确保图片大于 100x100)

    // 定义感兴趣区域 (ROI): 范围是 [行范围, 列范围]
    // 这里裁剪的是：高度 0~100，宽度 0~100
    Rect roi_rect(0, 0, 100, 100);
    Mat cropped_img = gray_img(roi_rect);

    // 显示裁剪图
    imshow("裁剪出的左上角", cropped_img);
    // 保存裁剪图
    imwrite("result_crop.jpg", cropped_img);
    cout << ">>> 左上角区域已裁剪并保存为 result_crop.jpg" << endl;

    // 输出某个像素值 (例如灰度图左上角 0,0 的像素值)
    // .at<uchar>(行, 列) 用于访问单通道灰度图像素
    cout << "左上角 (0,0) 的像素值: " << (int)gray_img.at<uchar>(0, 0) << endl;

    // --- 等待按键退出 ---
    cout << "按任意键退出程序..." << endl;
    waitKey(0);

    return 0;
}
