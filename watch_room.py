import pyrealsense2 as rs
import numpy as np
import cv2

# 配置深度和颜色流
pipeline = rs.pipeline()
config = rs.config()

# 获取设备产品线以实现更好的控制
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# 检查是否有找到的设备
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("本示例需要一个带颜色传感器的深度相机。")
    exit(0)

# 配置流的分辨率和帧率
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始串流
pipeline.start(config)

try:
    while True:

        # 等待一对连贯的帧: 深度和颜色
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 显示图像
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止串流
    pipeline.stop()
    cv2.destroyAllWindows()