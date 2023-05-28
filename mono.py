import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    input_img = 'test_imgs/p1.jpg'
    output_txt = input_img.split('.')[0] + '_mono_output.txt'
    
    mono(input_img, output_txt, equalize=True, gaussblur=True)
