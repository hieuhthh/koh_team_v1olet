FROM python:3.8

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get install gcc-8 g++-8

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Create working directory
WORKDIR /home/

COPY . .
  
CMD [ "python", "./detect.py" ]