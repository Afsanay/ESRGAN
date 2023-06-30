import os
import torch
import config
from torch import optim
from model import Generator
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title("ESRGAN")
    st.info("Change your perspective with Enhanced Super Resolution GAN")

st.title('Change low resolution images into high resolution')

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

gen = Generator(in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE, )
gen.eval()
options = os.listdir('LR')
selected_image = st.selectbox("Choose an image", options=options)

image_path = os.path.join("LR", selected_image)
if image_path is not None:
    image = Image.open(image_path)
    img = np.array(image)
    img = img[:, :, :3]
    image = Image.fromarray(img)
    with torch.no_grad():
        upscaled_img = gen(config.test_transform(image=np.asarray(image))["image"].unsqueeze(0).to(config.DEVICE))

    col1, col2 = st.columns(2)
    with col1:
        resized_image = image.resize((512, 512))
        st.info("Low Resolution Image")
        st.image(resized_image)

    with col2:
        y = torch.squeeze(upscaled_img).permute(1, 2, 0)
        final_arr = np.array(y.cpu())
        final_arr = final_arr[:, :, :3]
        final_image = Image.fromarray((final_arr * 255).astype(np.uint8))
        final_image = final_image.resize((512, 512))
        st.info("High Resolution Image")
        st.image(final_image)
