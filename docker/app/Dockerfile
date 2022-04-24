FROM python:3.8

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

RUN python -m pip install --upgrade pip

WORKDIR /usr/fine-tune/app

RUN git clone https://github.com/phamvanhanh6720/MLOps-HW.git /usr/fine-tune/app

RUN pip install streamlit
RUN pip install pillow

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]