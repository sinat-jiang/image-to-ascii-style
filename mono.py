"""
黑白图像 ascii 风格转换
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter


# CHARS = '@W#$OEXC[(/?=^~_.` '

# CHARS = '@W#$OEXC[]()/?=^~_.` '

CHARS = '@W#$OEXC=~_.` '            # recommend

# CHARS = "$,@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~\<\>i!lI;:,\^`'. "

# CHARS = '@%#*+=-:. '              # recommend【复杂背景推荐用这个】

# CHARS = '我爱你个大猪头@W#$OEXC[(/?=^~_.` '


def im2char(im, dsize):
    im = cv2.resize(im, dsize=dsize, interpolation=cv2.INTER_AREA)
    length = len(CHARS) - 1
    im = np.int32(np.round(im / 255 * length))
    output = []
    for y in range(dsize[1]):
        s = ""
        for x in range(dsize[0]):
            s += CHARS[im[y][x]]
        # print(s)
        output.append(s)
    # print(output)
    return '\n'.join(output)


def im2char_re_2darray(im, dsize):
    """
    返回 char array，based on im2char()
    """
    im = cv2.resize(im, dsize=dsize, interpolation=cv2.INTER_AREA)
    length = len(CHARS) - 1
    im = np.int32(np.round(im / 255 * length))
    output = []
    for y in range(dsize[1]):
        row = []
        for x in range(dsize[0]):
            row.append(CHARS[im[y][x]])
        output.append(row)
    return output


def get_background(
    choice: str, 
    origin,         # 原始图片输入
    width, 
    height, 
    glur: bool=False,
):
    """generate a canvas to print"""
    if choice == 'transparent':
        # 4-channel
        return Image.fromarray(np.uint8(np.zeros((height, width, 4))))
    elif choice == 'black':
        # 黑色蒙版
        return Image.fromarray(np.uint8(np.zeros((height, width, 3))))
    elif choice == 'white':
        img = Image.fromarray(np.uint8(np.ones((height, width, 3)) * 255))
        return img
    elif choice == 'red':
        bg_nda = np.concatenate([
            np.ones((height, width, 1)) * 255, np.zeros((height, width, 1)), np.zeros((height, width, 1))
        ], axis=-1) 
        img = Image.fromarray(np.uint8(bg_nda))
        return img
    elif choice == 'green':
        bg_nda = np.concatenate([
            np.zeros((height, width, 1)), np.ones((height, width, 1)) * 255, np.zeros((height, width, 1))
        ], axis=-1) 
        img = Image.fromarray(np.uint8(bg_nda))
        return img
    elif choice == 'blue':
        bg_nda = np.concatenate([
            np.zeros((height, width, 1)), np.zeros((height, width, 1)), np.ones((height, width, 1)) * 255
        ], axis=-1) 
        img = Image.fromarray(np.uint8(bg_nda))
        return img
    elif choice.startswith('customize'):
        color_rgb = [int(i) for i in choice.split('_')[-3:]]
        bg_nda = np.concatenate([
            np.ones((height, width, 1)) * color_rgb[0], np.ones((height, width, 1)) * color_rgb[1], np.ones((height, width, 1)) * color_rgb[2]
        ], axis=-1)
        img = Image.fromarray(np.uint8(bg_nda))
        return img
    elif choice == 'mean':
        mean = np.mean(np.array(origin)[:])
        return Image.fromarray(np.uint8(np.ones((height, width, 3)) * mean))
    elif choice.startswith('origin'):
        # 原图加上透明度
        opacity = float(choice[-1]) / 10
        canvas = origin.resize((width, height), Image.BICUBIC).filter(
            ImageFilter.GaussianBlur(15)
        )
        canvas = np.array(canvas)
        canvas = np.uint8(canvas[:, :, 0:3] * opacity)
        return Image.fromarray(canvas)
    

def mono(input: str, output: str = None, num_lines: int = 100, equalize: bool = False, gaussblur: bool = False):
    """
    原始函数，输出 txt 文件
    output grayscale .txt file
    """
    path = Path(input)
    im = cv2.imread(str(path))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(im, cmap='gray')
    
    # 直方图均衡化
    if equalize:
        im = cv2.equalizeHist(im)
        # plt.subplot(1,3,2)
        # plt.imshow(im, cmap='gray')
        
    # 加高斯模糊
    if gaussblur:
        im = cv2.GaussianBlur(im, ksize=(3,3), sigmaX=2, sigmaY=2)
        # plt.subplot(1,3,3)
        # plt.imshow(im, cmap='gray')
        
    # plt.show()

    height, width, *_ = im.shape
    output_height = num_lines
    output_width = round(width * 1.865 * output_height / height)
    # output_height = round(height / 1.865 * output_width / width)
    text = im2char(im, (output_width, output_height))
    
    if output is None:
        output = path.with_name(path.stem + '_mono_output.txt')
        
    with open(output, 'w') as f:
        f.write(text)
        print('write success')
        

def mono_ret_image(
    input: str, 
    output: str = None, 
    num_lines: int = 100, 
    equalize: bool = False, 
    gaussblur: bool = False,
    medianblur: bool = False,
    background: str = 'white',
    background_glur: bool = False,
    char_color: tuple = (0, 0, 0),
    out_height: int = None,
    scale: float = None,
    fontsize: int = 17,
    fonttype: str = None,
    hw_ratio: float = 1.865,
    char_width: float = 8.8,
):
    """
    改写原始 mono 函数，使其返回 image 而不是 text file
    
    Params:
        hw_ratio: 字符高宽比
        char_width: 画布中字符的宽（高则由 char_width * hw_ratio 得到）
    """
    input_path = Path(input)
    origin = cv2.imread(str(input_path))
    origin_for_bg = Image.open(input_path)
    origin = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
    height, width, *_ = origin.shape
    # print('image size:', height, width)
    
    # 直方图均衡化
    if equalize:
        # 1 全局直方图均衡化 (Global Histogram Equalization)
        origin = cv2.equalizeHist(origin)
        
        # 2 CLAHE 直方图均衡化
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # origin = clahe.apply(origin)
        
        # # 3 基于YCrCb色彩空间的直方图均衡化
        # # 转换到 YCrCb 色彩空间
        # ycrcb = cv2.cvtColor(cv2.imread(str(input_path)), cv2.COLOR_BGR2YCrCb)
        # # 对Y通道进行直方图均衡化
        # ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # # 转换回 BGR 色彩空间
        # origin = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        # origin = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
        
    # 加高斯模糊（可以实现降噪）
    if gaussblur:
        origin = cv2.GaussianBlur(origin, ksize=(3, 3), sigmaX=2, sigmaY=2)
        
    # 中值滤波，主要适用于以下几种图像噪音：椒盐噪声、横纹噪声、斑点噪声
    if medianblur:
        origin = cv2.medianBlur(origin, 5)  # 5 表示核的大小，可以根据需要调整
        
    # 输出尺寸计算
    output_height_rows = num_lines
    output_width_cols = round(width * hw_ratio * output_height_rows / height)    # 1.865 为所选字符的宽高比
    # print(output_height_rows, output_width_cols)    # (100, 280)
    
    # 获取输出文本
    text = im2char_re_2darray(origin, (output_width_cols, output_height_rows))
    # print(len(text), len(text[0]))      # (100, 280)
    
    text_rows, text_cols = output_height_rows, output_width_cols
    # 设置字体参数
    if fonttype == 'zh':
        # 中文字体
        font = ImageFont.truetype('simhei.ttf', fontsize, encoding='utf-8')
    else:
        font = ImageFont.truetype('courbd.ttf', fontsize)
    char_height = char_width * hw_ratio
    canvas_height = round(text_rows * char_height)
    canvas_width = round(text_cols * char_width)
    
    # 2 构建画布
    # a canvas used to draw texts on it
    canvas = get_background(background, origin_for_bg, canvas_width, canvas_height, background_glur)
    
    draw = ImageDraw.Draw(canvas)
    for i in range(text_cols):
        for j in range(text_rows):
            if fonttype == 'zh':        # 注意中文字符占两个字节
                x = round(char_width * 2 * i)
            else:
                x = round(char_width * i)
            y = round(char_height * j)
            char = text[j][i]
            color = char_color          # 字体颜色
            draw.text((x, y), char, fill=color, font=font)
    
    # resize the reproduct if necessary
    if out_height:                      # height goes first
        canvas_height = out_height
        canvas_width = round(width * canvas_height / height)
        canvas = canvas.resize((canvas_width, canvas_height), Image.BICUBIC)
    elif scale:
        canvas_width = round(width * scale)
        canvas_height = round(height * scale)
        canvas = canvas.resize((canvas_width, canvas_height), Image.BICUBIC)
        
    # output filename
    if output:
        output_path = Path(output)
    else:
        output_path = input_path.with_name(
            f'{input_path.stem}_{canvas_width}x{canvas_height}_D{text_rows}_{background}.png'
        )
    canvas.save(output_path)
    
    # print(f'Transformation completed. Saved as {output_path.name}.')

    

if __name__ == '__main__':
    # test for mono
    # input_img = 'test_imgs/p1.jpg'
    # output_txt = input_img.split('.')[0] + '_mono_output.txt'
    # mono(input_img, output_txt, equalize=True, gaussblur=True)
    
    # test for mono_ret_image
    # image = 'example/p1.jpg'
    # image = 'test_imgs/p182_c.jpg'
    # image = 'test_imgs/p165_c.jpg'
    image = 'test_imgs/p72_c.jpg'
    # image = 'test_imgs/p0_c.jpg'
    
    # 简单图像
    image = 'test_imgs/kuaishou/simple_images/3.jpg'
    # image = 'test_imgs/kuaishou/simple_images/p3_c.jpg'
    
    w, h = Image.open(image).convert('RGB').size
    output_image = f"{image.split('.')[0]}_mono_output.{image.split('.')[-1]}"
    
    kwargs = {
        'num_lines': 80,           # 字符行数，行数越大，细节越清晰
        # 'equalize': True,           # 直方图均衡化（对普通图像，即颜色分布较为均匀时，关闭直方图均衡化能获得较好的效果）
        'gaussblur': True,          # 高斯模糊
        'medianblur': False,        # 中值滤波
        'background': 'white',      # 背景版颜色
        # 'background': 'origin9',      # 背景版颜色
        # 'background': 'customize_17_238_238',  # 背景版颜色，可自定义三个通道的颜色
        'background_glur': False,   # 是否对背景板做模糊处理
        'fontsize': 17,             # 字体大小
        # 'fonttype': 'zh',         # 字体类型，区分中英文
        'char_color': (0, 0, 0),      # 字体颜色
        # 'char_color': (100,149,237),  # 字体颜色
        'out_height': None,         # 输出图片高度
        'char_width': 8.8,          # 字符宽度（和 fontsize 一起控制在底板上绘制字符的大小，char_width 越大，单个字符在底板上占据的空间就越大，此时若 fontsize 固定，则 width_size 越大，显示的字就越小）
    }
    mono_ret_image(image, output_image, **kwargs)
