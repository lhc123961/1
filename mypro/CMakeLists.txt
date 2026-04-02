cmake_minimum_required(VERSION 3.10)
project(mypro)

# --- 关键修改：使用 pkg-config ---
# 执行 pkg-config 命令，获取 OpenCV 的编译参数
execute_process(
    COMMAND pkg-config --cflags opencv4
    OUTPUT_VARIABLE OPENCV_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 执行 pkg-config 命令，获取 OpenCV 的链接库参数
execute_process(
    COMMAND pkg-config --libs opencv4
    OUTPUT_VARIABLE OPENCV_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 将获取到的编译参数应用到编译器
add_compile_options(${OPENCV_CFLAGS})

# --- 你的其他配置 ---
set(CMAKE_CXX_STANDARD 11)

# 添加可执行文件
add_executable(mypro sy2.cpp)  # 请确保文件名是 sy2.cpp，不是 sy2

# 将获取到的链接库参数应用到目标
target_link_libraries(mypro ${OPENCV_LIBS})