"""
对视频文件进行 mono 和 color 的 ascii 码图像风格转换
"""

import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm


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
    # print(im_list)
    im_list.sort(key=lambda x: int(x.split('.')[0].split('_')[2]))  # 最好再看看图片顺序对不
    img = Image.open(os.path.join(frames_path, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致
    # print('image size:', img_size)

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')      #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # opencv版本是3
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
    for i in tqdm(im_list):
        im_name = os.path.join(frames_path, i)
        # frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
    videoWriter.release()
    print('finish')


def frames_to_ascii(frames_path, args, type='mono'):
    """
    依次将每张图片做 ascii 风格转换
    
    Params: 
    - frames_path: 存放帧图像的路径 
    - args: 风格转换的参数 
    - type: 转换类型，黑白 or 彩色
    """
    pass



if __name__ == '__main__':
    # 拆帧
    video_path = f'./test_videos/bs.mp4'         # 视频文件路径
    output_folder = os.path.join(os.path.dirname(video_path), f"{os.path.basename(video_path).split('.')[0]}_frames")   # 输出文件夹路径
    video_frames_extract(video_path, output_folder, ff=None)

    # 组帧
    # v1
    # frames_path = r'D:\D1\AI_toys\CV\vision_transfor\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\photo2ascii_self\self_test_videos\pkq2_frames_ascii'
    # video_path = r'D:\D1\AI_toys\CV\vision_transfor\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\photo2ascii_self\self_test_videos\pkq2_ascii.mp4v'
    # frames_path = r'datasets\photo2ascii_self\self_test_videos\bs2_frames_ascii'
    # video_path = r'datasets\photo2ascii_self\self_test_videos\bs2_ascii.mp4v'
    # v2
    # frames_path = r'datasets\photo2ascii_self_size150_v2\self_test_videos\bs_frames_ascii'
    # video_path = r'datasets\photo2ascii_self_size150_v2\self_test_videos\bs_ascii.mp4v'
    # frames_path = r'datasets\photo2ascii_self_size150_v2\self_test_videos\pkq_frames_ascii'
    # video_path = r'datasets\photo2ascii_self_size150_v2\self_test_videos\pkq_ascii.mp4v'
    # frames_path = r'datasets\photo2ascii_self_size150_v2\self_test_videos\sl_frames_ascii'
    # video_path = r'datasets\photo2ascii_self_size150_v2\self_test_videos\sl_ascii.mp4v'
    # frames_path = r'bs_frames'
    # video_path = r'bs_ascii.mp4'
    # frames_to_video(frames_path, video_path, fps=59)
