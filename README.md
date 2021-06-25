**KO - Hackathon**

Participants:
Nguyen Mau Trong Hieu - Huynh Ngoc Tan

Docker run:
https://www.youtube.com/watch?v=GFRnjtjsfXA

Docker command:

sudo docker pull hieuhthh/kohfinal:third

sudo docker run hieuhthh/kohfinal:third

sudo docker run -it -v /home/hieu/video:/home/DETECT hieuhthh/kohfinal:third

Presentation:
https://drive.google.com/drive/u/0/folders/1eFawVAtl9BKvaz0e2NA-hUv5lBR9xZd9

We have inferred 2 videos:

Demo Video 1:
https://youtu.be/FgoyfaOtN_Y

Demo Video 2:
https://youtu.be/m7b-2d0ML5o

**HOW TO USE WITH GIT**

Quick Tutorial: https://youtu.be/EL_DgZFHHiI

Open command line

You need to change directory to this folder 

# Download library thing:

pip install -r requirements.txt

# Inference - To detect, copy and paste this line to command line:

python detect.py


Then write the path to video.
Example: 

E:\KOH\Robot_View\VID_20201222_141801.mp4

DETECT_THIS_FOLDER

**Result will be in runs\detect\exp**

**YOLO FORMAT**

**CLASSES IN LABELS**

0 - BOX

1 - PERSON

2 - FIRE

3 - DOG

4 - CAT

5 - ANOMALY (DROPPED BOX)

If DROPPED BOX is TRACKED, then it will be normal box.

**REFERENCE**

https://github.com/ultralytics/yolov5

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
