"""
黑白图像的一些封装操作
"""
from PIL import Image
import cv2
from pathlib import Path
from mono import mono, CHARS


def simple_mono(
    input: str, 
    output: str = None, 
    num_lines: int = 100, 
    equalize: bool = False, 
    gaussblur: bool = False
):
    """
    简化函数，直接输入 Image 类型，返回 Image 类型
    """
    pass


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
    