"""
黑白图像的一些封装操作
"""
from PIL import Image
import cv2
from pathlib import Path
from mono import mono, get_background, im2char_re_2darray
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def mono_imageio(
    input: Image, 
    num_lines: int = 100, 
    equalize: bool = False, 
    gaussblur: bool = False,
    background: str = 'white',
    background_glur: bool = False,
    char_color: tuple = (0, 0, 0),
    out_height: int = None,
    scale: float = None,
    fontsize: int = 17,
    hw_ratio: float = 1.865,
    char_width: float = 8.8,
):
    """
    修改版：直接输入 Image 类型，返回 Image 类型
    based on mono_ret_image() in mono.py
    """
    origin = cv2.cvtColor(np.asarray(input), cv2.COLOR_RGB2GRAY)
    height, width, *_ = origin.shape
    print('image size:', height, width)
    
    # 直方图均衡化
    if equalize:
        origin = cv2.equalizeHist(origin)
        
    # 加高斯模糊
    if gaussblur:
        origin = cv2.GaussianBlur(origin, ksize=(3,3), sigmaX=2, sigmaY=2)
        
    # 输出尺寸计算
    output_height_rows = num_lines
    output_width_cols = round(width * hw_ratio * output_height_rows / height)    # 1.865 为所选字符的宽高比
    
    # 获取输出文本
    text = im2char_re_2darray(origin, (output_width_cols, output_height_rows))
    
    text_rows, text_cols = output_height_rows, output_width_cols
    # 设置字体参数
    font = ImageFont.truetype('courbd.ttf', fontsize)
    char_height = char_width * hw_ratio
    canvas_height = round(text_rows * char_height)
    canvas_width = round(text_cols * char_width)
    
    # 2 构建画布
    # a canvas used to draw texts on it
    canvas = get_background(background, origin, canvas_width, canvas_height, background_glur)
    
    draw = ImageDraw.Draw(canvas)
    for i in range(text_cols):
        for j in range(text_rows):
            x = round(char_width * i)
            y = round(char_height * j)
            char = text[j][i]
            color = char_color       # 字体颜色
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
        
    return canvas


def auto_convert(input_image, output_image, **kwargs):
    """
    根据输入参数定制化生成结果
    :params input_image: 输入图片路径
    :params output_image: 输出图片路径
    """
    mono(
        input=input_image,
        output=output_image,
        **kwargs
    )    
    

if __name__ == '__main__':
    # single image test
    pass
    