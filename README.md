**KO - Hackathon**

Participants:
Nguyen Mau Trong Hieu - Huynh Ngoc Tan

We have inferred 2 videos:

Demo Video 1:
https://youtu.be/FgoyfaOtN_Y

Demo Video 2:
https://youtu.be/m7b-2d0ML5o

**HOW TO USE**

Open command line

You need to change directory to folder koh_team_v1olet 

Download library thing:

pip install -r requirements.txt

Put the video/image you want to detect into folder **DETECT_THIS_FOLDER** 

To detect, copy and pass one of these lines to command line:

# CPU inference

python detect.py --weights koh_model_detect_box_v2.pt --img 640 --conf 0.5 --iou 0.45 --save-txt --exist-ok --source DETECT_THIS_FOLDER

# GPU inference - faster - if you don't have the gpu or right driver version --> use CPU inference

python detect.py --weights koh_model_detect_box_v2.pt --img 640 --conf 0.5 --iou 0.45 --save-txt --exist-ok --source DETECT_THIS_FOLDER --device 0

**Result will be in runs\detect\exp**

**REFERENCE**

https://github.com/ultralytics/yolov5

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html


