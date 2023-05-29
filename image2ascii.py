import streamlit as st
from color_add_endwithop import simple_color
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.title('Image to ascii style~')
st.write('Convert the image to ascii style')

st.sidebar.write("## Upload and download")

uploaded_file = st.sidebar.file_uploader("", type=['jpg', 'png'], label_visibility='collapsed')

if uploaded_file is not None:
    image = uploaded_file
    st.image(image, caption='Origin Image')
        
    ascii_image = simple_color(Image.open(image))
    
    st.image(ascii_image, caption='The Ascii Style Image')
    
    # Download the fixed image
    def convert_image(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        return byte_im
    
    fixed = ascii_image
    st.sidebar.download_button("Download ascii image", convert_image(fixed), "ascii_image.png", "image/png")
    
else:
    col1, col2 = st.columns(2)
    col1.image('example/head.png', caption='Origin Image')
    col2.image('example/head_color_output.png', caption='The Ascii Style Image')
    
    