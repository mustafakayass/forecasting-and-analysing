FROM python:3.9.7

WORKDIR "/application"

COPY . .

RUN pip install --upgrade pip
RUN pip install -r /application/requirements.txt

ADD requirements.txt /application/
ADD deneme.py /application/
CMD [ "python", "deneme.py" ]




