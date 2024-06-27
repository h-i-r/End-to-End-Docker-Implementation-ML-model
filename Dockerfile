FROM python:3.11.7

COPY requirements.txt /purchase_predictor/requirements.txt
COPY src /purchase_predictor/src

WORKDIR /purchase_predictor

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /purchase_predictor/src

ENTRYPOINT ["python", "main.py"]

