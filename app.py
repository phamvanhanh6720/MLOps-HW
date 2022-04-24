import json
import base64
import requests
from PIL import Image

import streamlit as st

hostname = 'http://0.0.0.0:8000'

st.header('Image classification')
uploaded_file = st.file_uploader("Choose a file")
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=320)

        is_recognize = st.button('Classify')
    with col2:
        st.write("Kết quả:")

        if is_recognize:
            base64_img = base64.b64encode(bytes_data)
            base64_img = base64_img.decode('utf-8')

            url = '{}/classify'.format(hostname)
            data = {'base64_img': base64_img}

            response = requests.post(url=url, data=json.dumps(data))
            if response.status_code == 200:
                content = json.loads(response.content)
                class_name = content['class']
                prob = content['prob']

                st.write("{} with confidence {:.2f}%".format(class_name, prob * 100))

            else:
                st.write("Call API fail")