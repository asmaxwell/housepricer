# syntax=docker/dockerfile:1
# start by pulling the python image
FROM python:3.11-slim-buster

# switch working directory
WORKDIR /app

# copy the requirements file into the image
COPY ./requirements.txt /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

EXPOSE 5000
ENV FLASK_APP=app.py

# copy every content from the local file to the image
COPY . .


CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
