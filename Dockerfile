# syntax=docker/dockerfile:1
# start by pulling the python image
FROM python:3.11-slim-buster

# switch working directory
WORKDIR /src

# copy the requirements file into the image
COPY ./requirements.txt /src

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

EXPOSE 5000
ENV FLASK_APP=app/app.py

# copy all content from the local file to the image
COPY ./housepricer housepricer
COPY ./app app
# copy all content from the local file to the image
COPY ./data data

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
