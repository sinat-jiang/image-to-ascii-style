# ImageToText

字符画生成器

![](./demo.jpg)

## 安装教程

Python >= 3.7

```shell
pip install -r requirements.txt
```

## 使用说明

```shell
# 查看帮助
python main.py -- --help

# 黑白纯文本
python main.py mono -- --help
# 基本
python main.py mono input.jpg
# 设定输出名称和后缀
python main.py mono input.jpg --output output.png
# 设定文本行数
python main.py mono input.jpg --num_lones 125
# 直方图均值化（使图片对比度更高，解决整体偏亮或偏暗的问题）
python main.py mono input.jpg --equalize

# 彩色文本图片
python main.py color -- --help
# 基本
python main.py color input.jpg
# 设定输出
python main.py color input.jpg output.png
# 设定使用的字符
# uppercase, lowercase, alphabet, number, alphanumeric, symbol, random
python main.py color input.jpg --alphabet lowercase
# 设定背景
# transparent, black, white, mean, origin5, origin7, ...
# origin + n 表示背景为加了 nxn 高斯模糊的原图
python main.py color input.jpg --background black
# 设置输出图片的横向分辨率（高度）
python main.py color input.jpg --out_height 1200
# 设置输出图片的缩放（以原图尺寸为准）
# 如果同时设置了 out_height，则以 out_height 为准
python main.py color input.jpg --scale 1.25
```