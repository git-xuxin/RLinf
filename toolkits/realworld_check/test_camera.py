# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may in a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 查找所有连接的 RealSense 设备
    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        print("No RealSense devices found.")
        return

    # 为每个设备创建一个 pipeline 和 config
    pipelines = {}
    serial_numbers = []
    for device in devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"Found device: {serial_number}")
        serial_numbers.append(serial_number)

        pipeline = rs.pipeline(ctx)
        config = rs.config()
        
        # 将配置与特定设备序列号绑定
        config.enable_device(serial_number)
        config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8, # OpenCV 使用 BGR 格式
            30, # 帧率可以适当提高，如 30
        )
        pipelines[serial_number] = (pipeline, config)

    # 启动所有 pipeline
    for serial, (pipeline, config) in pipelines.items():
        try:
            pipeline.start(config)
            print(f"Pipeline started for device {serial}")
        except RuntimeError as e:
            print(f"Failed to start pipeline for device {serial}: {e}")
            # 如果某个设备启动失败，将其从字典中移除
            pipelines.pop(serial)

    if not pipelines:
        print("Could not start any camera pipelines.")
        return

    try:
        while True:
            # 遍历所有已启动的 pipeline
            for serial, (pipeline, _) in pipelines.items():
                # 等待帧，使用 poll_for_frames() 或 try_wait_for_frames() 避免阻塞
                frames = pipeline.poll_for_frames()
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # 将图像转换为 numpy 数组
                color_image = np.asanyarray(color_frame.get_data())
                
                # 创建一个唯一的窗口名称来显示图像
                window_name = f'RealSense Camera - {serial}'
                cv2.imshow(window_name, color_image)

            # 等待按键，如果按下 'q' 或 ESC 键则退出循环
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("'q' or ESC pressed, exiting...")
                break

    finally:
        # 确保在退出时停止所有 pipeline
        print("Stopping all pipelines...")
        for serial, (pipeline, _) in pipelines.items():
            pipeline.stop()
            print(f"Pipeline for device {serial} stopped.")
        cv2.destroyAllWindows()
        print("All OpenCV windows closed.")


if __name__ == "__main__":
    main()