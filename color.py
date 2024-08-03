import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import random
from pathlib import Path
import time
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt


"""
几组参数设置：
1）alphabet 为英文时，fontsize 取 17
2）alphabet 为中文时，fontsize 取 11
"""


def get_alphabet(choice):
    """get the alphabet used to print on the output image"""
    if choice == 'uppercase':
        return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    elif choice == 'lowercase':
        return 'abcdefghijklmnopqrstuvwxyz'
    elif choice == 'alphabet':
        return get_alphabet('uppercase') + get_alphabet('lowercase')
    elif choice == 'number':
        return '0123456789'
    elif choice == 'alphanumeric':
        return get_alphabet('alphabet') + get_alphabet('number')
    elif choice == 'symbol':
        return r'~!@#$%^&*()-_=+[]{}\|;:,<.>/?"'
    elif choice == 'random':
        return get_alphabet('alphanumeric') + get_alphabet('symbol')
    # 添加中文字体
    elif choice == 'number_zh_simple':
        return '一二三四五六七八九十'
    elif choice == 'number_zh_comp':
        return '壹贰叁肆伍陆柒捌玖拾'
    elif choice.startswith('sd_zh'):
        return choice.split('_')[-1]
    # 英文自定义
    elif choice == 'alphanumeric_en':
        return get_alphabet('alphabet') + get_alphabet('number')
    elif choice.startswith('sd_en'):
        return choice.split('_')[-1]


def get_background(choice: str, origin, width, height):
    """
    generate a canvas to print
    """
    if choice == 'transparent':
        # 4-channel
        return Image.fromarray(np.uint8(np.zeros((height, width, 4))))
    elif choice == 'black':
        return Image.fromarray(np.uint8(np.zeros((height, width, 3))))
    elif choice == 'white':
        return Image.fromarray(np.uint8(np.ones((height, width, 3)) * 255))
    elif choice == 'mean':
        mean = np.mean(np.array(origin)[:])
        return Image.fromarray(np.uint8(np.ones((height, width, 3)) * mean))
    elif choice.startswith('origin'):
        opacity = float(choice[-1]) / 10
        canvas = origin.resize((width, height), Image.BICUBIC).filter(
            ImageFilter.GaussianBlur(25)
        )
        canvas = np.array(canvas)
        canvas = np.uint8(canvas[:, :, 0:3] * opacity)
        return Image.fromarray(canvas)


def color(
    input: str,
    output: str = None,
    rows: int = 100,
    alphabet='uppercase',
    background='origin7',
    out_height: int = None,
    scale: float = None,
    fontsize: int = 17,
    hw_ratio: float = 1.25,
    char_width: float = 8.8,
    char_height: float = 11,        # = width * 1.25
    random_char: bool = True,
    char_width_gap_ratio: float = 1.0,
    char_height_gap_ratio: float = 1.0,
    zh_fonts: str='simhei.ttf',
):
    """
    output colorful text picture
    """
    input_path = Path(input)
    # the original image
    origin = Image.open(input_path)
    width, height = origin.size
    print(f'input size: {origin.size}')

    # text amount of the output image
    text_rows = rows
    text_cols = round(width / (height / text_rows) * hw_ratio)      # char height-width ratio
    origin_ref_np = cv2.resize(
        np.array(origin), (text_cols, text_rows), interpolation=cv2.INTER_AREA
    )
    origin_ref = Image.fromarray(origin_ref_np)
    
    # font properties
    if 'zh' in alphabet:
        # 中文可选字体类型（windows）：https://zhuanlan.zhihu.com/p/617230914
        font = ImageFont.truetype(zh_fonts, fontsize, encoding='utf-8')
    else:
        font = ImageFont.truetype('courbd.ttf', fontsize)

    # 调整 char 的宽高使得图片 size 尽量和原图一致
    # char_width = 8.8
    # char_height = 11        # = width * 1.25
    # output size depend on the rows and cols
    canvas_height = round(text_rows * char_height)
    canvas_width = round(text_cols * char_width)
    
    # a canvas used to draw texts on it
    if 'zh' in alphabet or 'en' in alphabet:    
        canvas = get_background(background, origin, int(canvas_width * char_width_gap_ratio), int(canvas_height * char_height_gap_ratio))
    else:
        canvas = get_background(background, origin, canvas_width, canvas_height)
    
    # start drawing
    since = time.time()
    print(f'Start transforming {input_path.name}')
    draw = ImageDraw.Draw(canvas)
    charlist = get_alphabet(alphabet)
    length = len(charlist)

    if not random_char:
        count = 0
        
    if 'zh' in alphabet:
        # 中文，调整输出顺序为从左至右，防止顺序展示时无法阅读
        for i in range(text_rows):
            for j in range(text_cols):
                x = round(char_width * char_width_gap_ratio * j)
                y = round(char_height * char_height_gap_ratio * i)
                if random_char:
                    char = charlist[random.randint(0, length - 1)]
                else:
                    char = charlist[count]
                    count = (count + 1) % len(charlist)
                color = origin_ref.getpixel((j, i))     # eg. (135, 82, 69)
                draw.text((x, y), char, fill=color, font=font)
    elif 'en' in alphabet:
        # 增加英文自定义功能
        for i in range(text_rows):
            for j in range(text_cols):
                x = round(char_width * char_width_gap_ratio * j)
                y = round(char_height * char_height_gap_ratio * i)
                if random_char:
                    char = charlist[random.randint(0, length - 1)]
                else:
                    char = charlist[count]
                    count = (count + 1) % len(charlist)
                color = origin_ref.getpixel((j, i))                 # eg. (135, 82, 69)
                draw.text((x, y), char, fill=color, font=font)
    else:
        # 原始写法（ascii 码字符，包括英文字母）
        for i in range(text_cols):
            for j in range(text_rows):
                x = round(char_width * i)
                y = round(char_height * j - 4)      # 减掉图像上面因为文字高度产生的白边
                if random_char:
                    char = charlist[random.randint(0, length - 1)]
                else:
                    char = charlist[count]
                    count = (count + 1) % len(charlist)
                color = origin_ref.getpixel((i, j))
                draw.text((x, y), char, fill=color, font=font)
    
    # plt.figure()
    # plt.imshow(draw)
    # plt.show()

    # resize the reproduct if necessary
    if out_height:  # height goes first
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

    print(f'Transformation completed. Saved as {output_path.name}.')
    print(f'Output image size: {canvas_width}x{canvas_height}')
    print(f'Text density: {text_cols}x{text_rows}')
    print(f'Elapsed time: {time.time() - since:.4} second(s)')


if __name__ == '__main__':

    # instance test
    # input_image = 'test_imgs/head.png'
    # output_image = input_image.split('.')[0] + '_color_output.jpg'

    # test for HD images
    input_image = 'test_imgs/kuaishou/simple_images/p23_c.jpg'
    # input_image = 'test_imgs/kuaishou/standard_images/7.jpg'
    output_image = f"{input_image.split('.')[0]}_color_output-200.{input_image.split('.')[-1]}"
    
    # 英文字符通用参数参考
    kwargs = {
        'rows': 200,
        # 'alphabet': 'uppercase',          # 字符填充类型
        'alphabet': 'alphanumeric',         # also you can use alphanumeric_en inorder to use the char_width_gap_ratio param to adjust the gap between chars
        # 'alphabet': 'sd_en_ I love you ',         # also you can use alphanumeric_en inorder to use the char_width_gap_ratio param to adjust the gap between chars
        'background': 'origin7',            # 背景色
        'out_height': None,
        'fontsize': 17,                     # 中文自定义需要手动调整字体大小已获得一个比较好的效果
        'hw_ratio': 1.25,
        'char_width': 8.8,
        'char_height': 11,                  # = width * 1.25
        'random_char': True,                # 是否将字符集随机分布在整张图片上
        'char_width_gap_ratio': 1.1,        # 字符间隔调整，防止拥挤
        'char_height_gap_ratio': 1.1,
    }

    # 中文字符参数参考
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
    #     'char_width_gap_ratio': 1.75,        # 中文字符间隔需要手动调整，防止拥挤
    #     'char_height_gap_ratio': 1.75,
    # }

    color(
        input=input_image,
        output=output_image,
        **kwargs
    )


