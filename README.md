**KO - Hackathon**

Participants:
Nguyen Mau Trong Hieu - Huynh Ngoc Tan

Presentation:
https://drive.google.com/file/d/15d23Md-SMNjTTmuoUCUrW_i-BVO-PIf2/view?usp=sharing

We have inferred 2 videos:

Demo Video 1:
https://youtu.be/FgoyfaOtN_Y

Demo Video 2:
https://youtu.be/m7b-2d0ML5o

**HOW TO USE**

Quick Tutorial: https://youtu.be/EL_DgZFHHiI

Open command line

You need to change directory to this folder 

# Download library thing:

pip install -r requirements.txt

To detect, copy and paste one of these lines to command line:

# Inference

python detect.py

Then write the path to video.

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


