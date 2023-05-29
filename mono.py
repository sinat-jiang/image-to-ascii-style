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
        
        
def get_background(choice: str, origin, width, height) -> Image.Image:
    """generate a canvas to print"""
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
    

def mono_ret_image(
    input: str, 
    output: str = None, 
    num_lines: int = 100, 
    equalize: bool = False, 
    gaussblur: bool = False
):
    """
    改写原始 mono 函数，使其返回 image 而不是 text file
    """
    input_path = Path(input)
    origin = cv2.imread(str(input_path))
    origin = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
    height, width, *_ = origin.shape
    print(height, width)
    
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(origin, cmap='gray')
    
    # 直方图均衡化
    if equalize:
        origin = cv2.equalizeHist(origin)
        plt.subplot(2, 3, 2)
        plt.imshow(origin, cmap='gray')
        
    # 加高斯模糊
    if gaussblur:
        origin = cv2.GaussianBlur(origin, ksize=(3,3), sigmaX=2, sigmaY=2)
        plt.subplot(2, 3, 3)
        plt.imshow(origin, cmap='gray')
        
    
    # 输出尺寸计算
    output_height_rows = num_lines
    output_width_cols = round(width * 1.865 * output_height_rows / height)    # 1.865 为所选字符的宽高比
    
    # 输出 txt 文件
    # text = im2char(origin, (output_width_cols, output_height_rows))
    # with open(output, 'w') as f:
    #     f.write(text)
    #     print('write success')
    
    # 输出 image 文件
    # 1 获取输出输出文本
    text = im2char_re_2darray(origin, (output_width_cols, output_height_rows))
    
    hw_ratio = 1.25
    text_rows = num_lines
    text_cols = round(width / (height / text_rows) * hw_ratio)  # char height-width ratio
    origin_ref_np = cv2.resize(
        np.array(origin), (text_cols, text_rows), interpolation=cv2.INTER_AREA
    )
    plt.subplot(2, 3, 4)
    plt.imshow(origin_ref_np, cmap='gray')
    
    fontsize = 17
    font = ImageFont.truetype('courbd.ttf', fontsize)
    char_width = 8.8
    char_height = 11        # = width * 1.25
    canvas_height = round(text_rows * char_height)
    canvas_width = round(text_cols * char_width)
    
    # 2 构建画布
    # a canvas used to draw texts on it
    canvas = get_background('white', origin, canvas_width, canvas_height)
    plt.subplot(2, 3, 5)
    plt.imshow(canvas)
    
    draw = ImageDraw.Draw(canvas)
    
    plt.show()
    

if __name__ == '__main__':
    # test
    # input_img = 'test_imgs/p1.jpg'
    # output_txt = input_img.split('.')[0] + '_mono_output.txt'
    # mono(input_img, output_txt, equalize=True, gaussblur=True)
    
    # test for mono_ret_image
    image = 'example/p1.jpg'
    output_image = image.split('.')[0] + '_mono_output.txt'
    kwargs = {
        'equalize': True,
        'gaussblur': True
    }
    mono_ret_image(image, output_image, **kwargs)
