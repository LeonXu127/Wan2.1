import cv2
import numpy as np
import os
import imageio

# 视频参数设置
width, height = 1280, 720
fps = 24
total_frames = 81
output_filename = 'motion_transfer/assets/moving_ellipse.mp4'
if os.path.exists(output_filename):
    os.remove(output_filename)

# 图形参数配置
shape = {
    'type': 'ellipse',    # 可选 'rectangle' 或 'ellipse'
    'color': (0, 255, 0), # BGR颜色格式（此处为绿色）
    'rect_size': (180, 360),  # 矩形尺寸（宽, 高）
    'ellipse_axes': (300, 300) # 椭圆轴长（长轴, 短轴）
}

# 计算初始位置和运动参数
if shape['type'] == 'rectangle':
    start_x = -shape['rect_size'][0]       # 从左侧外进入
    end_x = width
    y_pos = (height - shape['rect_size'][1]) // 2
elif shape['type'] == 'ellipse':
    # start_x = -shape['ellipse_axes'][0]
    # end_x = width + shape['ellipse_axes'][0]
    start_x = 0
    end_x = width
    y_pos = height // 2

step = (end_x - start_x) / (total_frames - 1)  # 移动步长计算

# 生成所有帧并保存为mp4
frames = []
for frame_num in range(total_frames):
    # 创建纯白色背景
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    # 计算当前X坐标
    current_x = int(start_x + frame_num * step)
    # 绘制图形
    if shape['type'] == 'rectangle':
        cv2.rectangle(frame, 
                     (current_x, y_pos),
                     (current_x + shape['rect_size'][0], y_pos + shape['rect_size'][1]),
                     shape['color'], -1)
    elif shape['type'] == 'ellipse':
        cv2.ellipse(frame, 
                   (current_x, y_pos),
                   shape['ellipse_axes'], 0, 0, 360, 
                   shape['color'], -1)
    # imageio要求RGB顺序
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

# 用imageio保存为mp4
imageio.mimsave(output_filename, frames, fps=fps)
print("视频生成完成！保存为:", output_filename)