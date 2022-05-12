# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
from PIL import Image
import ssl


# ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL
ssl._create_default_https_context = ssl._create_unverified_context

from functions_imageclassification import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def read_user_img(img_path):
    custom_dir = [img_path]

    # Turn custom data into batch
    custom_data = data_to_batch(custom_dir, test_data=True)
    # Make a prediction in custom image batch and returns label
    custom_predicted_label = predict_custom_image(custom_data)
    print("The model predicted the following.")
    print(custom_predicted_label)
    # Un-batch prediction- returns image file path
    custom_images = unbatchify(custom_data, has_labels=False)
    spanish_labels = translate_images(custom_images, custom_predicted_label)
    print("Spanish")
    print(spanish_labels)
    return spanish_labels, custom_predicted_label


def run():
    st.title("Fruits and vegetables- Image-based dictionary")
    # Let the user upload an image
    img_file = st.file_uploader("Choose and image", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        saved_image_path = './upload_images/' + img_file.name
        with open(saved_image_path, "wb") as handler:
            handler.write(img_file.getbuffer())

        if img_file is not None:
            spanish_label, predicted_label = read_user_img(saved_image_path)
            print(spanish_label)
            print(predicted_label)
            for i in range(0, len(predicted_label)):
                st.success("This seems to be a :" + predicted_label[i])
                st.info("Spanish: " + spanish_label[i])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
run()
