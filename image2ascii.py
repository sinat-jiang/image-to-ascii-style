import streamlit as st
from color_add_endwithop import color_imageio
from mono_add_endwithop import mono_imageio
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.title('Image to ascii style~')
st.write('Convert the image to ascii style, the higher the resolution of the picture, the better the effect.')
example_placeholder = st.empty()

st.sidebar.write("## Upload and download")
uploaded_file = st.sidebar.file_uploader("", type=['jpg', 'png'], label_visibility='collapsed')
download_placeholder = st.sidebar.empty()

st.sidebar.write("## Params setting")
image_type = st.sidebar.radio(
    "**Image Style:**",
    options=('monochrome', 'color'),
    horizontal = True,
)
if image_type == 'monochrome':
    # params setting
    background = st.sidebar.radio(
        "1 - Backgroud color:",
        options=('white', 'black'),
        horizontal=True,
    )
    color2rgb = {
        'black': (0, 0, 0), 
        'white': (255, 255, 255), 
        'red': (255, 0, 0), 
        'blue': (0, 0, 255), 
        'green': (0, 255, 0),
    }
    text_color = st.sidebar.selectbox(
        '2 - Text color:',
        options=color2rgb.keys(),
    )
    output_height_consistence = st.sidebar.radio(
        '3 - Same height as source image:',
        options=(False, True),
        horizontal=True,
    )
    equalize = st.sidebar.radio(
        '4 - Histogram equalization',
        options=(False, True),
        horizontal=True,
    )
    gaussblur = st.sidebar.radio(
        '5 - Gaussian blur',
        options=(False, True),
        horizontal=True,
    )
    output_char_rows = st.sidebar.radio(
        '6 - Rows of char in ascii image:',
        options=(70, 100, 130, 160),
        index=1,
        horizontal=True,
    )
else:
    alphabet = st.sidebar.radio(
        '1 - char set:',
        options=('uppercase',),
        horizontal=True,
    )
    background_type = st.sidebar.radio(
        '2 - background type:',
        options=('origin', 'white', 'black'),
        horizontal=True,
    )
    if background_type == 'origin':
        background_trans_level = st.sidebar.slider(
            label='Transparency level for backgroud:',
            min_value=0,
            max_value=9,
            value=7
        )
    output_height_consistence = st.sidebar.radio(
        '3 - Same height as source image:',
        options=(False, True),
        horizontal=True,
    )
    output_char_rows = st.sidebar.radio(
        '4 - Rows of char in ascii image:',
        options=(80, 100, 120, 140, 160),
        index=1,
        horizontal=True,
    )

if uploaded_file is not None:
    
    image = uploaded_file
    
    st.image(image, caption='Origin Image')
    
    if image_type == 'monochrome':
        kwargs = {
            'background': background,
            'char_color': color2rgb[text_color],
            'out_height': Image.open(image).size[1] if output_height_consistence else None,
            'equalize': equalize,
            'gaussblur': gaussblur,
            'num_lines': output_char_rows,
        }
        ascii_image = mono_imageio(Image.open(image), **kwargs)
    elif image_type == 'color':
        kwargs = {
            'alphabet': alphabet,
            'background': background_type+str(background_trans_level) if 'origin' in background_type else background_type,
            'out_height': Image.open(image).size[1] if output_height_consistence else None,
            'rows': output_char_rows,
        }
        ascii_image = color_imageio(Image.open(image), **kwargs)
    
    st.image(ascii_image, caption='The Ascii Style Image')
    
    # Download the fixed image
    def convert_image(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        return byte_im
    
    fixed = ascii_image
    download_placeholder.download_button("Download ascii image", convert_image(fixed), "ascii_image.png", "image/png")
    
else:
    example_placeholder.write('Example:')
    col1, col2, col3 = st.columns(3)
    col1.image('example/head.png', caption='Origin Image')
    col2.image('example/head_mono_output.png', caption='The Ascii Style Image(Mono)')
    col3.image('example/head_color_output.png', caption='The Ascii Style Image(Color)')
    
    