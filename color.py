# -*- encoding:utf-8 -*- 

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
    # 中文字符
    elif choice == 'number_zh_simple':
        return '一二三四五六七八九十'
    elif choice == 'number_zh_comp':
        return '壹贰叁肆伍陆柒捌玖拾'
    # 中文自定义
    elif choice.startswith('sd_zh'):
        return choice.split('_')[-1]
    # 英文自定义
    elif choice == 'alphanumeric_en':
        return get_alphabet('alphabet') + get_alphabet('number')
    elif choice.startswith('sd_en'):
        return choice.split('_')[-1]


def get_background(choice: str, origin, width, height, kernel=25):
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
    elif choice.startswith('grade'):
        return Image.fromarray(np.uint8(np.ones((height, width, 3)) * int(choice.split('_')[-1])))
    elif choice == 'mean':
        mean = np.mean(np.array(origin)[:])
        return Image.fromarray(np.uint8(np.ones((height, width, 3)) * mean))
    elif choice.startswith('origin'):
        opacity = float(choice[-1]) / 10
        canvas = origin.resize((width, height), Image.BICUBIC).filter(
            ImageFilter.GaussianBlur(kernel)
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
    kernel=5,
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
    font: str='courbd.ttf'
):
    """
    output colorful text picture
    """
    input_path = Path(input)
    # the original image
    origin = Image.open(input_path)
    width, height = origin.size
    # print(f'input size: {origin.size}')

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
        font = ImageFont.truetype(font, fontsize)

    # 调整 char 的宽高使得图片 size 尽量和原图一致
    # char_width = 8.8
    # char_height = 11        # = width * 1.25
    # output size depend on the rows and cols
    canvas_height = round(text_rows * char_height)
    canvas_width = round(text_cols * char_width)
    
    # a canvas used to draw texts on it
    canvas = get_background(background, origin, int(canvas_width * char_width_gap_ratio), int(canvas_height * char_height_gap_ratio), kernel)
    # if 'zh' in alphabet or 'en' in alphabet:    
    #     canvas = get_background(background, origin, int(canvas_width * char_width_gap_ratio), int(canvas_height * char_height_gap_ratio))
    # else:
    #     canvas = get_background(background, origin, canvas_width, canvas_height)
    
    # start drawing
    since = time.time()
    # print(f'Start transforming {input_path.name}')
    draw = ImageDraw.Draw(canvas)
    charlist = get_alphabet(alphabet)
    length = len(charlist)

    if not random_char:
        count = 0

    # 字符填充图片
    image_up_shift = 4              # 部分字体上部分会有额外的空隙，所以为了避免由此造成图像上部出现一段空白，需要将字符往上移动若干像素
    if 'zh' in alphabet:
        image_up_shift = 0
    for i in range(text_rows):
        for j in range(text_cols):
            x = round(char_width * char_width_gap_ratio * j)
            y = round(char_height * char_height_gap_ratio * i - image_up_shift)       
            if random_char:
                char = charlist[random.randint(0, length - 1)]
            else:
                char = charlist[count]
                count = (count + 1) % len(charlist)
            color = origin_ref.getpixel((j, i))     # eg. (135, 82, 69)
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

    # print(f'Transformation completed. Saved as {output_path.name}.')
    # print(f'Output image size: {canvas_width}x{canvas_height}')
    # print(f'Text density: {text_cols}x{text_rows}')
    # print(f'Elapsed time: {time.time() - since:.4} second(s)')


if __name__ == '__main__':

    # instance test
    # input_image = 'test_imgs/head.png'
    # output_image = input_image.split('.')[0] + '_color_output.jpg'

    # test for HD images
    # input_image = 'test_imgs/kuaishou/simple_images/p23_c.jpg'
    # input_image = 'test_imgs/kuaishou/standard_images/2.jpg'
    input_image = 'test_imgs/cangwentu/head9.jpg'
    output_image = f"{input_image.split('.')[0]}_color_output.{input_image.split('.')[-1]}"
    
    # 英文字符通用参数参考
    kwargs = {
        'rows': 150,
        # 'alphabet': 'uppercase',          # 字符填充类型
        # 'alphabet': 'alphanumeric',         # also you can use alphanumeric_en inorder to use the char_width_gap_ratio param to adjust the gap between chars
        'alphabet': "sd_en_Recording runtime status via logs is common for almost computer system, and detecting anomalies in logs is crucial for timely identifying malfunctions of systems. However, manually detecting anomalies for logs is time-consuming, error-prone, and infeasible. Existing automatic log anomaly detection approaches, using indexes rather than semantics of log templates, tend to cause false alarms. In this work, we propose LogAnomaly, a framework to model a log stream as a natural language sequence. Empowered by template2vec, a novel, simple yet effective method to extract the semantic information hidden in log templates, LogAnomaly can detect both sequential and quantitive log anomalies simultaneously, which has not been done by any previous work. Moreover, LogAnomaly can avoid the false alarms caused by the newly appearing log templates between periodic model retrainings. Our evaluation on two public production log datasets show that LogAnomaly outperforms existing logbased anomaly detection methods. The PostgreSQL Global Development Group announces that the second beta release of PostgreSQL 17 is now available for download. This release contains previews of all features that will be available when PostgreSQL 17 is made generally available, though some details of the release can change during the beta period. You can find information about all of the PostgreSQL 17 features and changes in the release notes. In the spirit of the open source PostgreSQL community, we strongly encourage you to test the new features of PostgreSQL 17 on your systems to help us eliminate bugs or other issues that may exist. While we do not advise you to run PostgreSQL 17 Beta 2 in production environments, we encourage you to find ways to run your typical application workloads against this beta release. Your testing and feedback will help the community ensure that the PostgreSQL 17 release upholds our standards of delivering a stable, reliable release of the world's most advanced open source relational database. Please read more about our beta testing process and how you can contribute. PostgreSQL 12 will stop receiving fixes on November 14, 2024. If you are running PostgreSQL 12 in a production environment, we suggest that you make plans to upgrade to a newer, supported version of PostgreSQL. Please see our versioning policy for more information.",         # also you can use alphanumeric_en inorder to use the char_width_gap_ratio param to adjust the gap between chars
        # 'alphabet': 'sd_en_miss you i miss you love you cherish you ',         # also you can use alphanumeric_en inorder to use the char_width_gap_ratio param to adjust the gap between chars
        # 'background': 'origin7',            # 背景色，默认为 origin7，数字表示不透明度，可以用来控制图片亮度
        'background': 'grade_100',            # 背景色，grade_100，纯色蒙版，数字表示蒙版的颜色数值，可以用来控制图片亮度
        # 'background': 'transparent',            # 如果是 png 图片，可以使用透明蒙版
        'kernel': 5,                        # 背景蒙版的高斯核大小，控制图片清晰度
        'out_height': None,
        'fontsize': 57,                     # 字体，默认 17
        'hw_ratio': 1.25,                   # 高宽比，默认 1.25
        'char_width': 8.8,                  # 字符宽度，默认 8.8
        'char_height': 11,                  # 字符高度，默认 = width * 1.25 = 11
        'random_char': False,               # 是否将字符集随机分布在整张图片上
        'char_width_gap_ratio': 3.7,        # 字符间隔调整，防止拥挤，默认 1.1
        'char_height_gap_ratio': 3.7,
        # 'font': 'segoesc.ttf',              # 偏手写风格字体
        'font': 'Inkfree.ttf',              # 偏手写风格字体
    }

    # 中文字符参数参考
    # kwargs = {
    #     'rows': 150,
    #     # 'alphabet': 'sd_zh_永和九年，岁在癸丑，暮春之初，会于会稽山阴之兰亭，修禊事也。群贤毕至，少长咸集。此地有崇山峻岭，茂林修竹；又有清流激湍，映带左右，引以为流觞曲水，列坐其次。虽无丝竹管弦之盛，一觞一咏，亦足以畅叙幽情。是日也，天朗气清，惠风和畅。仰观宇宙之大，俯察品类之盛，所以游目骋怀，足以极视听之娱，信可乐也。夫人之相与，俯仰一世，或取诸怀抱，悟言一室之内；或因寄所托，放浪形骸之外。虽趣舍万殊，静躁不同，当其欣于所遇，暂得于己，快然自足，不知老之将至。及其所之既倦，情随事迁，感慨系之矣。向之所欣，俯仰之间，已为陈迹，犹不能不以之兴怀。况修短随化，终期于尽。古人云：“死生亦大矣。”岂不痛哉！每览昔人兴感之由，若合一契，未尝不临文嗟悼，不能喻之于怀。固知一死生为虚诞，齐彭殇为妄作。后之视今，亦犹今之视昔。悲夫！故列叙时人，录其所述，虽世殊事异，所以兴怀，其致一也。后之览者，亦将有感于斯文。',     # 字符填充类型
    #     'alphabet': 'sd_zh_北冥有鱼，其名为鲲。鲲之大，不知其几千里也；化而为鸟，其名为鹏。鹏之背，不知其几千里也；怒而飞，其翼若垂天之云。是鸟也，海运则将徙于南冥。南冥者，天池也。《齐谐》者，志怪者也。《谐》之言曰：“鹏之徙于南冥也，水击三千里，抟扶摇而上者九万里，去以六月息者也。”野马也，尘埃也，生物之以息相吹也。天之苍苍，其正色邪？其远而无所至极邪？其视下也，亦若是则已矣。且夫水之积也不厚，则其负大舟也无力。覆杯水于坳堂之上，则芥为之舟，置杯焉则胶，水浅而舟大也。风之积也不厚，则其负大翼也无力。故九万里，则风斯在下矣，而后乃今培风；背负青天，而莫之夭阏者，而后乃今将图南。蜩与学鸠笑之曰：“我决起而飞，抢榆枋而止，时则不至，而控于地而已矣，奚以之九万里而南为？”适莽苍者，三餐而反，腹犹果然；适百里者，宿舂粮；适千里者，三月聚粮。之二虫又何知！小知不及大知，小年不及大年。奚以知其然也？朝菌不知晦朔，蟪蛄不知春秋，此小年也。楚之南有冥灵者，以五百岁为春，五百岁为秋；上古有大椿者，以八千岁为春，八千岁为秋，此大年也。而彭祖乃今以久特闻，众人匹之，不亦悲乎！汤之问棘也是已。穷发之北，有冥海者，天池也。有鱼焉，其广数千里，未有知其修者，其名为鲲。有鸟焉，其名为鹏，背若泰山，翼若垂天之云，抟扶摇羊角而上者九万里，绝云气，负青天，然后图南，且适南冥也。斥鴳笑之曰：“彼且奚适也？我腾跃而上，不过数仞而下，翱翔蓬蒿之间，此亦飞之至也。而彼且奚适也？”此小大之辩也。故夫知效一官，行比一乡，德合一君，而征一国者，其自视也，亦若此矣。而宋荣子犹然笑之。且举世誉之而不加劝，举世非之而不加沮，定乎内外之分，辩乎荣辱之境，斯已矣。彼其于世，未数数然也。虽然，犹有未树也。夫列子御风而行，泠然善也，旬有五日而后反。彼于致福者，未数数然也。此虽免乎行，犹有所待者也。若夫乘天地之正，而御六气之辩，以游无穷者，彼且恶乎待哉？故曰：至人无己，神人无功，圣人无名。',     # 字符填充类型
    #     # 'alphabet': 'number_zh_comp',
    #     # 'alphabet': 'sd_zh_我踏马裂开~',         # 字符填充类型
    #     # 'alphabet': 'sd_zh_真香', 
    #     # 'background': 'origin7',            # 背景色，默认为 origin7，数字表示不透明度，可以用来控制图片亮度
    #     'background': 'grade_20',            # 背景色，默认为 origin7，数字表示不透明度，可以用来控制图片亮度
    #     'kernel': 5,                        # 背景蒙版的高斯核大小，控制图片清晰度
    #     'out_height': None,
    #     'fontsize': 27,                     # 中文自定义需要手动调整字体大小已获得一个比较好的效果，默认 17
    #     'hw_ratio': 1.25,
    #     'char_width': 8.8,
    #     'char_height': 11,                  # = width * 1.25
    #     'random_char': False,               # 是否将字符集随机分布在整张图片上
    #     'char_width_gap_ratio': 2.75,        # 中文字符间隔需要手动调整，防止拥挤，默认 1.75
    #     'char_height_gap_ratio': 2.75,
    # }

    color(
        input=input_image,
        output=output_image,
        **kwargs
    )


