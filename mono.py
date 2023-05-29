import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter


CHARS = '@W#$OEXC[(/?=^~_.` '


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


def mono(input: str, output: str = None, num_lines: int = 100, equalize: bool = False, gaussblur: bool = False):
    """output grayscale .txt file"""
    path = Path(input)
    im = cv2.imread(str(path))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(im, cmap='gray')
    
    # 直方图均衡化
    if equalize:
        im = cv2.equalizeHist(im)
        plt.subplot(1,3,2)
        plt.imshow(im, cmap='gray')
        
    # 加高斯模糊
    if gaussblur:
        im = cv2.GaussianBlur(im, ksize=(3,3), sigmaX=2, sigmaY=2)
        plt.subplot(1,3,3)
        plt.imshow(im, cmap='gray')
        
    plt.show()

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
        
        
def get_background(choice: str, origin, width, height, glur=False) -> Image.Image:
    """generate a canvas to print"""
    if choice == 'transparent':
        # 4-channel
        return Image.fromarray(np.uint8(np.zeros((height, width, 4))))
    elif choice == 'black':
        # 黑色蒙版
        return Image.fromarray(np.uint8(np.zeros((height, width, 3))))
    elif choice == 'white':
        img = Image.fromarray(np.uint8(np.ones((height, width, 3)) * 255))
        # 白色蒙版
        if glur:
            img.filter(
                ImageFilter.GaussianBlur(25)
            )
        return img
    elif choice == 'mean':
        mean = np.mean(np.array(origin)[:])
        return Image.fromarray(np.uint8(np.ones((height, width, 3)) * mean))
    elif choice.startswith('origin'):
        # 原图加上透明度
        opacity = float(choice[-1]) / 10
        canvas = origin.resize((width, height), Image.BICUBIC).filter(
            ImageFilter.GaussianBlur(25)
        )
        canvas = np.array(canvas)
        canvas = np.uint8(canvas[:, :, 0:3] * opacity)
        return Image.fromarray(canvas)
    

def mono_ret_image(
    input: str, 
    output: str = None, 
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
    改写原始 mono 函数，使其返回 image 而不是 text file
    :params hw_ratio: 字符高宽比
    :params char_width: 画布中字符的宽（高则由 char_width * hw_ratio 得到）
    """
    input_path = Path(input)
    origin = cv2.imread(str(input_path))
    origin = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
    height, width, *_ = origin.shape
    print('image size:', height, width)
    
    # plt.figure()
    # plt.subplot(2, 3, 1)
    # plt.imshow(origin, cmap='gray')
    
    # 直方图均衡化
    if equalize:
        origin = cv2.equalizeHist(origin)
        # plt.subplot(2, 3, 2)
        # plt.imshow(origin, cmap='gray')
        
    # 加高斯模糊
    if gaussblur:
        origin = cv2.GaussianBlur(origin, ksize=(3,3), sigmaX=2, sigmaY=2)
        # plt.subplot(2, 3, 3)
        # plt.imshow(origin, cmap='gray')
        
    # 输出尺寸计算
    output_height_rows = num_lines
    output_width_cols = round(width * hw_ratio * output_height_rows / height)    # 1.865 为所选字符的宽高比
    # print(output_height_rows, output_width_cols)    # (100, 280)
    
    # 获取输出文本
    text = im2char_re_2darray(origin, (output_width_cols, output_height_rows))
    # print(len(text), len(text[0]))      # (100, 280)
    
    text_rows, text_cols = output_height_rows, output_width_cols
    # 设置字体参数
    font = ImageFont.truetype('courbd.ttf', fontsize)
    char_height = char_width * hw_ratio
    canvas_height = round(text_rows * char_height)
    canvas_width = round(text_cols * char_width)
    
    # 2 构建画布
    # a canvas used to draw texts on it
    canvas = get_background(background, origin, canvas_width, canvas_height, background_glur)
    # plt.subplot(2, 3, 5)
    # plt.imshow(canvas)
    
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
        
    # output filename
    if output:
        output_path = Path(output)
    else:
        output_path = input_path.with_name(
            f'{input_path.stem}_{canvas_width}x{canvas_height}_D{text_rows}_{background}.png'
        )
    canvas.save(output_path)
    
    # plt.subplot(2, 3, 5)
    # plt.imshow(canvas)
    
    # plt.show()
    
    print(f'Transformation completed. Saved as {output_path.name}.')

    

if __name__ == '__main__':
    # test
    # input_img = 'test_imgs/p1.jpg'
    # output_txt = input_img.split('.')[0] + '_mono_output.txt'
    # mono(input_img, output_txt, equalize=True, gaussblur=True)
    
    # test for mono_ret_image
    # image = 'example/p1.jpg'
    # image = 'test_imgs/p182_c.jpg'
    image = 'test_imgs/p0_c.jpg'
    w, h = Image.open(image).convert('RGB').size
    output_image = image.split('.')[0] + '_mono_output.jpg'
    kwargs = {
        'equalize': True,
        'gaussblur': True,
        'fontsize': 17,             # 字体大小
        'background': 'white',      # 背景版颜色
        'background_glur': False,    # 是否对背景板做模糊处理
        'char_color': (0, 0, 0),    # 字体颜色
        'out_height': None,         # 输出图片高度
        'char_width': 8.8,          # 字符宽度（和 fontsize 一起控制在底板上绘制字符的大小，char_width 越大，单个字符在底板上占据的空间就越大，此时若 fontsize 固定，则 width_size 越大，显示的字就越小）
    }
    mono_ret_image(image, output_image, **kwargs)
