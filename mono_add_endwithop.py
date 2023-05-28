"""
黑白图像的一些封装操作
"""
from PIL import Image


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
    