FROM python:3.8

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

RUN python -m pip install --upgrade pip

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /usr/fine-tune/api

RUN git clone https://github.com/phamvanhanh6720/MLOps-HW.git /usr/fine-tune/api

# COPY . /usr/fine-tune/api/

RUN ls

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "run:api", "--port", "8000", "--host", "0.0.0.0"]