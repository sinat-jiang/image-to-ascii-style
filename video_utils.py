"""
对视频文件进行 mono 和 color 的 ascii 码图像风格转换
"""

import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from mono import mono_ret_image
from color import color
from concurrent.futures import ProcessPoolExecutor


def video_frames_extract(video_path, output_folder, ff=None):
    """
    拆帧
    :param video_path:
    :param output_folder:
    :param fps:
    :return:
    """
    # 获取视频文件名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 新建文件夹
    output_path = output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    # 获取&设置帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('视频默认帧率：', fps)
    if ff is None:
        # 设置帧间隔为 1，即每帧的图片都会被抽取出来
        frame_interval = 1
    else:
        frame_interval = int(ff)
    print('设置帧间隔：', frame_interval)

    # 逐帧提取并保存
    pbar = tqdm(desc='抽取帧数：')
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % frame_interval == 0:
                image_name = os.path.join(output_path, f"{video_name}_frame_{count}.jpg")
                cv2.imwrite(image_name, frame)
            count += 1
            pbar.update(1)
        else:
            break
    cap.release()
    print('video split success.')
    return fps


def frames_to_video(frames_path, video_path, fps=25):
    """
    组帧
    :param frames_path:
    :param video_path:
    :param fps: 每秒多少帧，推荐和原视频保持一致
    :return:
    """
    im_list = os.listdir(frames_path)
    im_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))    # 对帧图片进行排序
    img = Image.open(os.path.join(frames_path, im_list[0]))
    img_size = img.size                                 # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致
    # print('image size:', img_size)

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')        # opencv 版本是 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')            # opencv 版本是 3
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
    for i in tqdm(im_list, desc='video synthesizing:'):
        im_name = os.path.join(frames_path, i)
        # frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
    videoWriter.release()
    print('video synthesize finished.')


def frames_to_ascii(frames_path, new_frames_save_path, kwargs, type='mono', max_workers=20):
    """
    依次将每张图片做 ascii 风格转换
    
    Params: 
    - frames_path: 存放帧图像的路径 
    - new_frames_save_path: 转换后的图像存储路径
    - args: 风格转换的参数 
    - type: 转换类型，黑白 or 彩色
    """

    if not os.path.exists(new_frames_save_path):
        os.makedirs(new_frames_save_path)

    # 读取所有文件
    frames = os.listdir(frames_path)
    # frames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))    # 对帧图片进行排序
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for frame in tqdm(frames, desc='Image2ascii:'):
            frame = os.path.join(frames_path, frame)
            # 依次进行转换并存储
            output_image = os.path.join(new_frames_save_path, os.path.basename(frame))
            if type == 'mono':
                task = pool.submit(mono_ret_image, frame, output_image, **kwargs)
                # mono_ret_image(frame, output_image, **kwargs)
            elif type == 'color':
                task = pool.submit(color, frame, output_image, **kwargs)
                # color(frame, output_image, **kwargs)
            else:
                print('请指定正确的转换类型')
    
    print('All frames are converted to ascii style.')


if __name__ == '__main__':
    # 拆帧
    video_path = f'./test_videos/regeng/regeng_ikun.mp4'         # 视频文件路径
    output_folder = os.path.join(os.path.dirname(video_path), f"{os.path.basename(video_path).split('.')[0]}_frames")   # 输出文件夹路径
    # fps = video_frames_extract(video_path, output_folder, ff=None)

    # Ascii 风格转化
    type = 'mono'      # mono or color
    new_frames_save_path = os.path.join(os.path.dirname(output_folder), f"{os.path.basename(video_path).split('.')[0]}_{type}_ascii_frames")
    # params for mono
    kwargs = {
        'num_lines': 60,           # 字符行数，行数越大，细节越清晰
        # 'equalize': True,         # 直方图均衡化（对普通图像，即颜色分布较为均匀时，关闭直方图均衡化能获得较好的效果）
        'gaussblur': True,          # 高斯模糊
        'medianblur': False,        # 中值滤波
        'background': 'white',      # 背景版颜色，也可以自定义，格式：'customize_17_238_238'
        'background_glur': False,   # 是否对背景板做模糊处理
        'fontsize': 17,             # 字体大小
        'char_color': (0, 0, 0),    # 字体颜色，可定制，例如：(100,149,237)
        'out_height': None,         # 输出图片高度
        'char_width': 8.8,          # 字符宽度（和 fontsize 一起控制在底板上绘制字符的大小，char_width 越大，单个字符在底板上占据的空间就越大，此时若 fontsize 固定，则 width_size 越大，显示的字就越小）
    }
    # params for color-en
    # kwargs = {
    #     'rows': 80,
    #     'alphabet': 'alphanumeric',               # 字符填充：大小写字母 + 数字
    #     'background': 'origin7',            # 背景色
    #     'out_height': None,
    #     'fontsize': 17,                     # 中文自定义需要手动调整字体大小已获得一个比较好的效果
    #     'hw_ratio': 1.25,
    #     'char_width': 8.8,
    #     'char_height': 11,                  # = width * 1.25
    #     'random_char': True,                # 是否将字符集随机分布在整张图片上
    #     'char_width_gap_ratio': 1.1,        # 中文字符间隔需要手动调整，防止拥挤
    #     'char_height_gap_ratio': 1.1,
    # }
    # params for color-zh
    # kwargs = {
    #     'rows': 80,
    #     # 'alphabet': 'sd_zh_我是大帅比〇',    # 字符填充类型
    #     # 'alphabet': 'number_zh_comp',
    #     # 'alphabet': 'sd_zh_我踏马裂开~',         # 字符填充类型
    #     'alphabet': 'sd_zh_真香', 
    #     'background': 'origin7',            # 背景色，数字表示不透明度，可以用来控制图片亮度
    #     'out_height': None,
    #     'fontsize': 17,                     # 中文自定义需要手动调整字体大小已获得一个比较好的效果
    #     'hw_ratio': 1.25,
    #     'char_width': 8.8,
    #     'char_height': 11,                  # = width * 1.25
    #     'random_char': False,               # 是否将字符集随机分布在整张图片上
    #     'char_width_gap_ratio': 1.75,       # 中文字符间隔需要手动调整，防止拥挤
    #     'char_height_gap_ratio': 1.75,
    # }
    # frames_to_ascii(frames_path=output_folder, new_frames_save_path=new_frames_save_path, kwargs=kwargs, type=type, max_workers=5)

    # 组帧
    fps = 30
    video_path = os.path.join(os.path.dirname(video_path), os.path.basename(video_path).replace('.', f'2ascii_{type}.'))
    frames_to_video(frames_path=new_frames_save_path, video_path=video_path, fps=fps)
