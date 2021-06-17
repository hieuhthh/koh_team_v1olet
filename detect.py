import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import os
import numpy as np
import pandas as pd
import shutil
import bisect

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def Dis(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

# Bisect
IMSIZE = 224
tan = 5.2
bias = [[0, 0, 0, IMSIZE], [IMSIZE, 0, IMSIZE, IMSIZE],
        [0, 0, IMSIZE, 0], [0, IMSIZE, IMSIZE, IMSIZE],
        [0, 0, IMSIZE, IMSIZE], [0, IMSIZE, IMSIZE, 0]]

def intersect(lines):
    p0, p1 = lines[:, :2], lines[:, 2:]
    v = (p1-p0) / np.linalg.norm(p1-p0, axis=1)[:, np.newaxis]

    projs = np.eye(v.shape[1]) - v[:, :, np.newaxis] @ v[:, np.newaxis, :]
    R = projs.sum(axis=0)

    q = (projs @ p0[:,:,np.newaxis]).sum(axis=0)

    p = np.linalg.lstsq(R, q, rcond=None)[0]
    
    return tuple(int(x) for x in p[:, 0])

def detect_lines(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = cv2.Canny(src, 50, 400, apertureSize = 3)
    pts = cv2.HoughLinesP(src, 1, np.pi/180, 75, minLineLength=10, maxLineGap=100)
    if pts is None:
        return
    return pts[:, 0, :], src

def classify(lines):
    lines = np.vstack((lines, bias))

    vertical_lines, horizontal_lines, slanting_lines = [], [], []

    for p in lines:
        x1, y1, x2, y2 = (int(x) for x in p)
        slope = abs((y1-y2)/(x1-x2+1e-6))
        if slope > 1e1: 
            vertical_lines.append(p)
        elif slope < 5e-1:
            horizontal_lines.append(p)
        else:   
            slanting_lines.append(p)

    return np.array(vertical_lines), np.array(horizontal_lines), np.array(slanting_lines)

def find_keyzone(center, v, h):

    def deviation(mid, arr):
        arr = np.hstack((arr[:, 0], arr[:, 2], [mid//2, mid-1, mid+1, (mid+IMSIZE)//2]*2))

        arr = np.sort(arr)
        
        x = bisect.bisect_left(arr, mid)
        
        lo, hi = np.median(arr[:x][-7:]), np.median(arr[x:][:7])
        return lo, hi

    l, r = deviation(center[0], v)
    u, d = deviation(center[1], h)

    return tuple(int(x) for x in (l, u, r, d))

def draw_lines(src, lines):
    for p in lines:
        x1, y1, x2, y2 = (int(x) for x in p)
        cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), IMSIZE//256)

def draw(src, center, keyzone):
    cv2.circle(src, center, IMSIZE//64, (255, 0 , 0), -1)
    
    x1, y1, x2, y2 = (int(x) for x in keyzone)
    cv2.rectangle(src, (x1, y1), (x2, y2), (255, 0, 0), IMSIZE//64)

def window(src):
    try:
    
        src = cv2.resize(src.copy(), (IMSIZE, IMSIZE))

        lines, dst = detect_lines(src.copy())

        #classify into 3 sets vertical, horizontal, and slanting lines
        vertical_lines, horizontal_lines, slanting_lines = classify(lines)

        #find vanishing point 
        vanishing_point = intersect(slanting_lines)

        #find keyzone
        keyzone = find_keyzone(vanishing_point, vertical_lines, horizontal_lines)

        return vanishing_point, keyzone
    
    except Exception as error:
        # return (112, 112), (112, 112, 112, 112)
        return None, None



# Detect

def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path('DETECT') / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[0, 255, 0], [255, 0, 0], [204, 204, 0], [178, 102, 255], [255, 102, 178], [204, 229, 255], [0, 0, 255], [187, 174, 252]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    preAno = []
    curAno = []
    corner = 0
    
    for path, img, im0s, vid_cap in dataset:


        if corner > 0:
            corner -= 1
            
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                    
            continue
        
        
        image = img.copy()
        image = np.transpose(image, (1, 2, 0))
        vanishing_point, keyzone = window(image)
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
                
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    if int(cls) != 0: # not box
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}' if opt.save_conf_plt else f'{names[c]}'
                            plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=2)
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    else: # box
                        
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        
                        if abs(xywh[0] - 0.5) < 0.3 and xywh[1] < 0.6:
                            continue
                        
                        anomalyBox = False
                   
                        
                        if keyzone is None:
                            if abs(xywh[0] - 0.5) < 0.07:
                                anomalyBox = True
                        else:
                            xl = xywh[1] / tan + keyzone[0] / 224.0 + 0.035
                            xr = keyzone[2] / 224.0 - xywh[1] / tan - 0.035
        
                            if xywh[0] > xl and xywh[0] < xr:
                                anomalyBox = True
        
        
                        if anomalyBox == True:
                        
                            curBox = xywh.copy()
                            curAno.append(curBox)
        
                            tracked = False
            
                            for preBox in preAno:
                                if Dis(curBox, preBox) < 0.2:
                                    tracked = True
                                    break
                            
                            if tracked:
                                if save_txt:  # Write to file
                                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or opt.save_crop or view_img:  # Add bbox to image
                                    c = 7  # integer class
                                    label = f'Tracked {conf:.2f}' if opt.save_conf_plt else f'Tracked'
                                    plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=2)
                                    if opt.save_crop:
                                        save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                                
                            else: 
                                
                                if save_txt:  # Write to file
                                    line = (5., *xywh, conf) if opt.save_conf else (5., *xywh)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or opt.save_crop or view_img:  # Add bbox to image
                                    c = 6  # integer class
                                    label = f'DroppedBox {conf:.2f}' if opt.save_conf_plt else 'DroppedBox'
                                    plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=2)
                                    if opt.save_crop:
                                        save_one_box(xyxy, im0s, file=save_dir / 'crops' / 'DroppedBox' / f'{p.stem}.jpg', BGR=True)
                                        
                        else:
                            if save_txt:  # Write to file
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or opt.save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = f'{names[c]} {conf:.2f}' if opt.save_conf_plt else f'{names[c]}'
                                plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=2)
                                if opt.save_crop:
                                    save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
               
               
            # no det == corner
            else: 
                corner = 30               
            # Reset Save Ano Box
            preAno = curAno.copy()
            curAno = []          


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='DETECT', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=True, action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-conf-plt', default=False, action='store_true', help='plot image with confidences')
    
    opt = parser.parse_args()
    
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
