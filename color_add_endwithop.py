"""
添加一些自定义 or 后处理操作
"""
from color import color, get_alphabet, get_background
from PIL import Image
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import random
import numpy as np
import time
from video_utils import video_frames_extract, frames_to_video
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil


def endwith_resize(input_image, output_image, **kwargs):
    """[todo]
    对输出进行 resize 操作，保证图像像素尺寸尽量一致
    """
    w, h = Image.open(input_image).size
    kwargs['out_height'] = h
    color(
        input=input_image,
        output=output_image,
        **kwargs
    )
    # resize
    pass
    

def auto_convert(input_image, output_image, **kwargs):
    """
    根据输入参数定制化生成结果
    :params input_image: 输入图片路径
    :params output_image: 输出图片路径
    """
    color(
        input=input_image,
        output=output_image,
        **kwargs
    )


def simple_color(
    input: Image,
    rows: int = 100,
    alphabet='uppercase',
    background='origin7',
    out_height: int = None,
    scale: float = None,
    fontsize: int = 17,
):
    """
    简化函数，直接输入 Image 类型，返回 Image 类型
    """
    origin = input
    width, height = origin.size
    # print(f'input size: {origin.size}')
    
    # text amount of the output image
    hw_ratio = 1.25
    text_rows = rows
    text_cols = round(width / (height / text_rows) * hw_ratio)  # char height-width ratio
    origin_ref_np = cv2.resize(
        np.array(origin), (text_cols, text_rows), interpolation=cv2.INTER_AREA
    )
    origin_ref = Image.fromarray(origin_ref_np)
    
    # font properties
    if 'zh' in alphabet:
        font = ImageFont.truetype('simhei.ttf', fontsize, encoding='utf-8')
    else:
        font = ImageFont.truetype('courbd.ttf', fontsize)
    # 调整 char 的宽高使得图片 size 尽量和原图一致
    char_width = 8.8
    char_height = 11        # = width * 1.25
    # char_width = width / text_cols
    # char_height = height / text_rows
    # output size depend on the rows and cols
    canvas_height = round(text_rows * char_height)
    canvas_width = round(text_cols * char_width)
    
    # a canvas used to draw texts on it
    canvas = get_background(background, origin, canvas_width, canvas_height)
    
    # start drawing
    since = time.time()
    print(f'Start transforming ...')
    draw = ImageDraw.Draw(canvas)
    charlist = get_alphabet(alphabet)
    length = len(charlist)

    for i in range(text_cols):
        for j in range(text_rows):
            x = round(char_width * i)
            y = round(char_height * j - 4)
            char = charlist[random.randint(0, length - 1)]
            color = origin_ref.getpixel((i, j))
            draw.text((x, y), char, fill=color, font=font)
    # resize the reproduct if necessary
    if out_height:  # height goes first
        canvas_height = out_height
        canvas_width = round(width * canvas_height / height)
        canvas = canvas.resize((canvas_width, canvas_height), Image.BICUBIC)
    elif scale:
        canvas_width = round(width * scale)
        canvas_height = round(height * scale)
        canvas = canvas.resize((canvas_width, canvas_height), Image.BICUBIC)
        
    print(f'Elapsed time: {time.time() - since:.4} second(s)')

    return canvas


def video_color(video_path, video_output_path, ff=1, rbfps=None):
    """
    对视频进行 color ascii style 风格转换
    """
    # 1 临时文件夹，存储拆帧结果
    output_folder='tmp'
    # 2 拆帧
    fps = video_frames_extract(video_path, output_folder, ff)
    # 3 转换
    images = os.listdir(output_folder)
    images.sort(key=lambda x: int(x.split('.')[0].split('_')[2]))
    # 进程池
    with ProcessPoolExecutor(max_workers=50) as pool:
        for img in tqdm(images, desc='convert to ascii style ...'):
            img = os.path.join(output_folder, img)
            task = pool.submit(auto_convert, img, img)
    # 4 组帧
    frames_to_video(output_folder, video_output_path, fps)
    # 5 删除临时文件夹
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    
        

if __name__ == '__main__':
    # instance test
    # input_image = 'test_imgs/p180_c.jpg'
    # output_image = input_image.split('.')[0] + '_color_output.jpg'
    # kwargs = {
    #     'rows':100, 
    #     # 'out_height':1103, 
    #     # 'alphabet':'number_zh_simple',
    #     # 'alphabet':'sd_zh_泰裤啦~',
    #     # 'fontsize': 10,
    # }
    # auto_convert(input_image, output_image, **kwargs)
    
    # video test
    video_path = r'D:\D1\AI_toys\CV\vision_transfor\AsciiStyleImageGan\videos\pkq.mp4'
    video_output_path = video_path.split('.')[0] + '_ascii.mp4'
    video_color(video_path, video_output_path, ff=1)
    