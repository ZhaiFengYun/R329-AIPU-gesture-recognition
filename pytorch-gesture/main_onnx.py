#################
#@zhai fengyun -->Yanshan University
#################
import numpy
import numpy as np
import cv2
import os
from typing import Tuple
import io
import sys
sys.path.append('/mnt/data/tsm-cannymotion2')
#
# import tvm
# import tvm.relay
import time
import onnx
import torch
# import torchvision
import torch.onnx
# from PIL import Image, ImageOps
# import tvm.contrib.graph_runtime as graph_runtime
from online_demo.tsm_1 import MobileNetV2
import onnxruntime as onnrun

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True


def image_transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

    image = image[16:240, 16:240, :]

    image = numpy.array(image, dtype='float32')

    cv2.normalize(image, image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)

    image = image.transpose(2, 0, 1)

    image[0, :, :] = (image[0, :, :] - 0.485) / 0.229
    image[1, :, :] = (image[1, :, :] - 0.456) / 0.224
    image[2, :, :] = (image[2, :, :] - 0.406) / 0.225

    image = numpy.array([image], dtype='float32')

    #print(image.max(),image.min())

    return image


catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]

n_still_frame = 0


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


WINDOW_NAME = 'Video Gesture Recognition'

WINDOW_NAME2 = 'Image Show'
his_i=[]
his_i.append(2)
his_i.append(2)
def main():
    global his_i
    x0a = numpy.zeros([1, 3, 56, 56], dtype='float32')
    x1a = numpy.zeros([1, 4, 28, 28], dtype='float32')
    x2a = numpy.zeros([1, 4, 28, 28], dtype='float32')
    x3a = numpy.zeros([1, 8, 14, 14], dtype='float32')
    x4a = numpy.zeros([1, 8, 14, 14], dtype='float32')
    x5a = numpy.zeros([1, 8, 14, 14], dtype='float32')
    x6a = numpy.zeros([1, 12, 14, 14], dtype='float32')
    x7a = numpy.zeros([1, 12, 14, 14], dtype='float32')
    x8a = numpy.zeros([1, 20, 7, 7], dtype='float32')
    x9a = numpy.zeros([1, 20, 7, 7], dtype='float32')


    image_dir=os.listdir('/mnt/data/myimage')

    print(image_dir)




    # x0a, x1a, x2a, x3a, x4a, x5a, x6a, x7a, x8a, x9a = x0.numpy(), x1.numpy(), x2.numpy(), x3.numpy(), x4.numpy(), x5.numpy(), x6.numpy(), x7.numpy(), x8.numpy(), x9.numpy()
    session = onnrun.InferenceSession('/mnt/data/tsm-cannymotion2/online_demo/mobilenetv2_jester_online.onnx')


    input_name = session.get_inputs()
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    print(cap)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    cv2.namedWindow(WINDOW_NAME2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME2, 640, 480)
    cv2.moveWindow(WINDOW_NAME2, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME2, WINDOW_NAME2)

    t = None
    index = 0

    idx = 0
    history = [2]
    history_logit = []
    history_timing = []

    i_frame = -1


    j=3

    print("Ready!")
    while True:
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()
            input_var = image_transform(img)

            feat, x0a, x1a, x2a, x3a, x4a, x5a, x6a, x7a, x8a, x9a = session.run([], {input_name[0].name: input_var,
                                                                                      input_name[1].name: x0a,
                                                                                      input_name[2].name: x1a,
                                                                                      input_name[3].name: x2a,
                                                                                      input_name[4].name: x3a,
                                                                                      input_name[5].name: x4a,
                                                                                      input_name[6].name: x5a,
                                                                                      input_name[7].name: x6a,
                                                                                      input_name[8].name: x7a,
                                                                                      input_name[9].name: x8a,
                                                                                      input_name[10].name: x9a,

                                                                                      })


            # feat, shift_buffer = executor(input_var, shift_buffer)
            if SOFTMAX_THRES > 0:
                feat_np = feat.reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat, axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat, axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat)
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

            his_i.append(idx)

            if(len(his_i)>2):
                his_i=his_i[-2:]

            ims = cv2.imread('/mnt/data/myimage/' + image_dir[j])
            ims = cv2.resize(ims, (640, 480))
            ims = cv2.resize(ims, (640, 480))


            cv2.imshow(WINDOW_NAME2, ims)

            if(his_i[0]==2):
                if (idx == 16):
                    j = j + 1
                    if (j > len(image_dir) - 1):
                        j = 0
                    ims = cv2.imread('/mnt/data/myimage/' + image_dir[j])
                    ims = cv2.resize(ims, (640, 480))
                    cv2.imshow(WINDOW_NAME2, ims)

                if (idx == 17):
                    j = j - 1
                    if (j < 0):
                        j = 3
                    ims = cv2.imread('/mnt/data/myimage/' + image_dir[j])
                    ims = cv2.resize(ims, (640, 480))
                    cv2.imshow(WINDOW_NAME2, ims)

            t2 = time.time()
            #print(f"{index} {catigories[idx]}")

            current_time = t2 - t1

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()


main()
