import numpy
import numpy as np
import cv2
import os
from typing import Tuple
import io
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
import sys

sys.path.append('/mnt/data/tsm-cannymotion2')
from online_demo.mobilenet_v2_tsm import MobileNetV2

addr1 = '/mnt/data/aipubuild/tf50/resnet_50_model/resnet_v1_50_frozen.pb'
addr2 = '/mnt/data/tsm-cannymotion2/online_demo/model.pb'
addr3 = '/mnt/data/tf_tsm/model3.pb'
import tensorflow as tf

SOFTMAX_THRES = 0
HISTORY_LOGIT = False
REFINE_OUTPUT = True


def image_transform(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

    image = image[16:240, 16:240, :]

    image = numpy.array(image, dtype='float32')

    cv2.normalize(image, image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)

    # image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    # image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    # image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
    # image[:, :, 0] = (image[:, :, 0] - 127)
    # image[:, :, 1] = (image[:, :, 1] - 127)
    # image[:, :, 2] = (image[:, :, 2] -127)

    image = numpy.array([image], dtype='float32')
    #print(image)

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
        print('add')
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

# addr_str='/mnt/data/tf_tsm/node.txt'
# with open(addr_str, "r") as f:
#     data = f.readlines()
# res=data[0]
# res=res.strip('[')
# res=res.strip(']')
# res=res.split(',')
model_nodes = ['Conv/Conv2D:0', 'BatchNorm/FusedBatchNormV3:0', 'Relu6:0', 'SeparableConv2d/depthwise:0',
               'BatchNorm_1/FusedBatchNormV3:0', 'Relu6_1:0', 'Conv_1/Conv2D:0', 'BatchNorm_2/FusedBatchNormV3:0',
               'Conv_2/Conv2D:0', 'BatchNorm_3/FusedBatchNormV3:0', 'Relu6_2:0', 'SeparableConv2d_1/depthwise:0',
               'BatchNorm_4/FusedBatchNormV3:0', 'Relu6_3:0', 'Conv_3/Conv2D:0', 'BatchNorm_5/FusedBatchNormV3:0',
               'Slice:0', 'Slice_1:0', 'concat:0', 'Conv_4/Conv2D:0', 'BatchNorm_6/FusedBatchNormV3:0', 'Relu6_4:0',
               'SeparableConv2d_2/depthwise:0', 'BatchNorm_7/FusedBatchNormV3:0', 'Relu6_5:0', 'Conv_5/Conv2D:0',
               'BatchNorm_8/FusedBatchNormV3:0', 'Add:0', 'Conv_6/Conv2D:0', 'BatchNorm_9/FusedBatchNormV3:0',
               'Relu6_6:0', 'SeparableConv2d_3/depthwise:0', 'BatchNorm_10/FusedBatchNormV3:0', 'Relu6_7:0',
               'Conv_7/Conv2D:0', 'BatchNorm_11/FusedBatchNormV3:0', 'Slice_2:0', 'Slice_3:0', 'concat_1:0',
               'Conv_8/Conv2D:0', 'BatchNorm_12/FusedBatchNormV3:0', 'Relu6_8:0', 'SeparableConv2d_4/depthwise:0',
               'BatchNorm_13/FusedBatchNormV3:0', 'Relu6_9:0', 'Conv_9/Conv2D:0', 'BatchNorm_14/FusedBatchNormV3:0',
               'Add_1:0', 'Slice_4:0', 'Slice_5:0', 'concat_2:0', 'Conv_10/Conv2D:0', 'BatchNorm_15/FusedBatchNormV3:0',
               'Relu6_10:0', 'SeparableConv2d_5/depthwise:0', 'BatchNorm_16/FusedBatchNormV3:0', 'Relu6_11:0',
               'Conv_11/Conv2D:0', 'BatchNorm_17/FusedBatchNormV3:0', 'Add_2:0', 'Conv_12/Conv2D:0',
               'BatchNorm_18/FusedBatchNormV3:0', 'Relu6_12:0', 'SeparableConv2d_6/depthwise:0',
               'BatchNorm_19/FusedBatchNormV3:0', 'Relu6_13:0', 'Conv_13/Conv2D:0', 'BatchNorm_20/FusedBatchNormV3:0',
               'Slice_6:0', 'Slice_7:0', 'concat_3:0', 'Conv_14/Conv2D:0', 'BatchNorm_21/FusedBatchNormV3:0',
               'Relu6_14:0', 'SeparableConv2d_7/depthwise:0', 'BatchNorm_22/FusedBatchNormV3:0', 'Relu6_15:0',
               'Conv_15/Conv2D:0', 'BatchNorm_23/FusedBatchNormV3:0', 'Add_3:0', 'Slice_8:0', 'Slice_9:0', 'concat_4:0',
               'Conv_16/Conv2D:0', 'BatchNorm_24/FusedBatchNormV3:0', 'Relu6_16:0', 'SeparableConv2d_8/depthwise:0',
               'BatchNorm_25/FusedBatchNormV3:0', 'Relu6_17:0', 'Conv_17/Conv2D:0', 'BatchNorm_26/FusedBatchNormV3:0',
               'Add_4:0', 'Slice_10:0', 'Slice_11:0', 'concat_5:0', 'Conv_18/Conv2D:0',
               'BatchNorm_27/FusedBatchNormV3:0', 'Relu6_18:0', 'SeparableConv2d_9/depthwise:0',
               'BatchNorm_28/FusedBatchNormV3:0', 'Relu6_19:0', 'Conv_19/Conv2D:0', 'BatchNorm_29/FusedBatchNormV3:0',
               'Add_5:0', 'Conv_20/Conv2D:0', 'BatchNorm_30/FusedBatchNormV3:0', 'Relu6_20:0',
               'SeparableConv2d_10/depthwise:0', 'BatchNorm_31/FusedBatchNormV3:0', 'Relu6_21:0', 'Conv_21/Conv2D:0',
               'BatchNorm_32/FusedBatchNormV3:0', 'Slice_12:0', 'Slice_13:0', 'concat_6:0', 'Conv_22/Conv2D:0',
               'BatchNorm_33/FusedBatchNormV3:0', 'Relu6_22:0', 'SeparableConv2d_11/depthwise:0',
               'BatchNorm_34/FusedBatchNormV3:0', 'Relu6_23:0', 'Conv_23/Conv2D:0', 'BatchNorm_35/FusedBatchNormV3:0',
               'Add_6:0', 'Slice_14:0', 'Slice_15:0', 'concat_7:0', 'Conv_24/Conv2D:0',
               'BatchNorm_36/FusedBatchNormV3:0', 'Relu6_24:0', 'SeparableConv2d_12/depthwise:0',
               'BatchNorm_37/FusedBatchNormV3:0', 'Relu6_25:0', 'Conv_25/Conv2D:0', 'BatchNorm_38/FusedBatchNormV3:0',
               'Add_7:0', 'Conv_26/Conv2D:0', 'BatchNorm_39/FusedBatchNormV3:0', 'Relu6_26:0',
               'SeparableConv2d_13/depthwise:0', 'BatchNorm_40/FusedBatchNormV3:0', 'Relu6_27:0', 'Conv_27/Conv2D:0',
               'BatchNorm_41/FusedBatchNormV3:0', 'Slice_16:0', 'Slice_17:0', 'concat_8:0', 'Conv_28/Conv2D:0',
               'BatchNorm_42/FusedBatchNormV3:0', 'Relu6_28:0', 'SeparableConv2d_14/depthwise:0',
               'BatchNorm_43/FusedBatchNormV3:0', 'Relu6_29:0', 'Conv_29/Conv2D:0', 'BatchNorm_44/FusedBatchNormV3:0',
               'Add_8:0', 'Slice_18:0', 'Slice_19:0', 'concat_9:0', 'Conv_30/Conv2D:0',
               'BatchNorm_45/FusedBatchNormV3:0', 'Relu6_30:0', 'SeparableConv2d_15/depthwise:0',
               'BatchNorm_46/FusedBatchNormV3:0', 'Relu6_31:0', 'Conv_31/Conv2D:0', 'BatchNorm_47/FusedBatchNormV3:0',
               'Add_9:0', 'Conv_32/Conv2D:0', 'BatchNorm_48/FusedBatchNormV3:0', 'Relu6_32:0',
               'SeparableConv2d_16/depthwise:0', 'BatchNorm_49/FusedBatchNormV3:0', 'Relu6_33:0', 'Conv_33/Conv2D:0',
               'BatchNorm_50/FusedBatchNormV3:0', 'Conv_34/Conv2D:0', 'BatchNorm_51/FusedBatchNormV3:0', 'Relu6_34:0',
               'fully_connected/BiasAdd:0']
mode2=['Slice_1/begin:0',
 'Slice_1/begin:0',
 'Slice_3/begin:0',
 'Slice_5/begin:0',
 'Slice_7/begin:0',
 'Slice_9/begin:0',
 'Slice_11/begin:0',
 'Slice_13/begin:0',
 'Slice_15/begin:0',
 'Slice_17/begin:0',
 'Slice_19/begin:0',
 'Mean:0',
 'Mean_1:0',
 'Slice/begin:0',
 'Slice_2/begin:0',
 'Slice_4/begin:0',
'Slice_6/begin:0',
 'Slice_8/begin:0',
 'Slice_10/begin:0',
 'Slice_12/begin:0',
 'Slice_14/begin:0',
 'Slice_16/begin:0',
 'Slice_18/begin:0']
for i in mode2:
    print(i)
    model_nodes.append(i)

inpu_nodes = ['x_input:0', 'x_input0:0', 'x_input1:0', 'x_input2:0', 'x_input3:0', 'x_input4:0',
                          'x_input5:0', 'x_input6:0', 'x_input7:0', 'x_input8:0', 'x_input9:0']

for i in inpu_nodes:
    print(i)
    model_nodes.append(i)

def main():
    x0a = numpy.zeros([1, 56, 56, 3], dtype='float32')
    x1a = numpy.zeros([1, 28, 28, 4], dtype='float32')
    x2a = numpy.zeros([1, 28, 28, 4], dtype='float32')
    x3a = numpy.zeros([1, 14, 14, 8], dtype='float32')
    x4a = numpy.zeros([1, 14, 14, 8], dtype='float32')
    x5a = numpy.zeros([1, 14, 14, 8], dtype='float32')
    x6a = numpy.zeros([1, 14, 14, 12], dtype='float32')
    x7a = numpy.zeros([1, 14, 14, 12], dtype='float32')
    x8a = numpy.zeros([1, 7, 7, 20], dtype='float32')
    x9a = numpy.zeros([1, 7, 7, 20], dtype='float32')

    # x0a, x1a, x2a, x3a, x4a, x5a, x6a, x7a, x8a, x9a = x0.numpy(), x1.numpy(), x2.numpy(), x3.numpy(), x4.numpy(), x5.numpy(), x6.numpy(), x7.numpy(), x8.numpy(), x9.numpy()

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

    t = None
    index = 0

    idx = 0
    history = [2]
    history_logit = []
    history_timing = []

    i_frame = -1

    print("Ready!")
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(addr3, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            # init = tf.initialize_all_variables()
            sess.run(init)

            # print all ops, check input/output tensor name.
            # uncomment it if you donnot know io tensor names.

            # print('-------------ops---------------------')
            # op = sess.graph.get_operations()
            #
            # for m in op:
            #
            #     print(m.values())
            #
            #
            #
            # print('-------------ops done.---------------------')
            #
            #
            # print(ldcs)

            input_x = sess.graph.get_tensor_by_name("x_input:0")  # input

            input0 = sess.graph.get_tensor_by_name("x_input0:0")  # input
            input1 = sess.graph.get_tensor_by_name("x_input1:0")  # input
            input2 = sess.graph.get_tensor_by_name("x_input2:0")  # input]
            input3 = sess.graph.get_tensor_by_name("x_input3:0")  # input

            input4 = sess.graph.get_tensor_by_name("x_input4:0")  # input
            input5 = sess.graph.get_tensor_by_name("x_input5:0")  # input]
            input6 = sess.graph.get_tensor_by_name("x_input6:0")  # input

            input7 = sess.graph.get_tensor_by_name("x_input7:0")  # input
            input8 = sess.graph.get_tensor_by_name("x_input8:0")  # input]
            input9 = sess.graph.get_tensor_by_name("x_input9:0")  # input

            # outputs1 = sess.graph.get_tensor_by_name('fully_connected/BiasAdd:0')
            #
            # outputs2 = sess.graph.get_tensor_by_name('Slice:0')
            # outputs3 = sess.graph.get_tensor_by_name('Slice_2:0')
            # outputs4 = sess.graph.get_tensor_by_name('Slice_4:0')
            # outputs5 = sess.graph.get_tensor_by_name('Slice_6:0')
            # outputs6 = sess.graph.get_tensor_by_name('Slice_8:0')
            # outputs7 = sess.graph.get_tensor_by_name('Slice_10:0')
            # outputs8 = sess.graph.get_tensor_by_name('Slice_12:0')
            # outputs9 = sess.graph.get_tensor_by_name('Slice_14:0')
            # outputs10 = sess.graph.get_tensor_by_name('Slice_16:0')
            # outputs11 = sess.graph.get_tensor_by_name('Slice_18:0')



            nodes_out = [sess.graph.get_tensor_by_name(i) for i in model_nodes]

            # inpu_nodes = ['x_input:0', 'x_input0:0', 'x_input1:0', 'x_input2:0', 'x_input3:0', 'x_input4:0',
            #               'x_input5:0', 'x_input6:0', 'x_input7:0', 'x_input8:0', 'x_input9:0']

            print(nodes_out)

            max_dict = dict()
            min_dict = dict()


            for i in range(len(inpu_nodes)):
                max_dict[inpu_nodes[i][:-2] + '_0'] = 0
                min_dict[inpu_nodes[i][:-2] + '_0'] = 0

            for i in range(len(model_nodes)):
                max_dict[model_nodes[i][:-2] + '_0'] = 0
                min_dict[model_nodes[i][:-2] + '_0'] = 0

            p_max_av = 0
            p_min_av = 0

            inputd_max = 0
            inputd_min = 0

            # nodes=[sess.graph.get_tensor_by_name(i[1:-1]) for i in res]
            # print(nodes)
            file_dirs=os.listdir('/mnt/data/20bn-jester-v1')
            for name in file_dirs:

                x0a = numpy.zeros([1, 56, 56, 3], dtype='float32')
                x1a = numpy.zeros([1, 28, 28, 4], dtype='float32')
                x2a = numpy.zeros([1, 28, 28, 4], dtype='float32')
                x3a = numpy.zeros([1, 14, 14, 8], dtype='float32')
                x4a = numpy.zeros([1, 14, 14, 8], dtype='float32')
                x5a = numpy.zeros([1, 14, 14, 8], dtype='float32')
                x6a = numpy.zeros([1, 14, 14, 12], dtype='float32')
                x7a = numpy.zeros([1, 14, 14, 12], dtype='float32')
                x8a = numpy.zeros([1, 7, 7, 20], dtype='float32')
                x9a = numpy.zeros([1, 7, 7, 20], dtype='float32')



                addr='/mnt/data/20bn-jester-v1/'+name

                file=sorted(os.listdir(addr))
                print(file)

                for fil_name in file:

                    image_addr=addr+'/'+fil_name
                    print(image_addr)


                #for rt in range(1):
                    # while True:
                    i_frame += 1
                    img=cv2.imread(image_addr)
                    #_, img = cap.read()  # (480, 640, 3) 0 ~ 255
                    if i_frame % 1 == 0:  # skip every other frame to obtain a suitable frame rate
                        t1 = time.time()
                        input_var = image_transform(img)

                        feat0 = \
                            sess.run(
                                nodes_out,

                                feed_dict={
                                    input_x: input_var,
                                    input0: x0a,
                                    input1: x1a,
                                    input2: x2a,

                                    input3: x3a,
                                    input4: x4a,
                                    input5: x5a,

                                    input6: x6a,
                                    input7: x7a,
                                    input8: x8a,

                                    input9: x9a

                                })

                        for i in range(len(model_nodes)):
                            p_max_av = feat0[i].max()
                            p_min_av = feat0[i].min()
                            # if 'Slice' in model_nodes[i]:
                            #     print(model_nodes[i],'  ',i,feat0[i].shape)
                            # if 'full' in  model_nodes[i]:
                            #     print(model_nodes[i],i)

                            max_dict[model_nodes[i][:-2] + '_0'] = np.array(
                                max(max_dict[model_nodes[i][:-2] + '_0'], p_max_av))
                            min_dict[model_nodes[i][:-2] + '_0'] = np.array(
                                min(min_dict[model_nodes[i][:-2] + '_0'], p_min_av))

                        x0a = feat0[16]
                        x1a = feat0[36]
                        x2a = feat0[48]

                        x3a = feat0[68]
                        x4a = feat0[80]
                        x5a = feat0[92]

                        x6a = feat0[112]
                        x7a = feat0[124]
                        x8a = feat0[144]

                        x9a = feat0[156]

                        feat = feat0[179]

                        # print(model_nodes[i], feat0[i].max(), feat0[i].min())

                        # print(feat0[0].shape,feat0[1].shape)

                        # if SOFTMAX_THRES > 0:
                        #     feat_np = feat.reshape(-1)
                        #     feat_np -= feat_np.max()
                        #     softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
                        #
                        #     print(max(softmax))
                        #     if max(softmax) > SOFTMAX_THRES:
                        #         idx_ = np.argmax(feat, axis=1)[0]
                        #     else:
                        #         idx_ = idx
                        # else:
                        #     idx_ = np.argmax(feat, axis=1)[0]

                        # if HISTORY_LOGIT:
                        #     history_logit.append(feat)
                        #     history_logit = history_logit[-12:]
                        #     avg_logit = sum(history_logit)
                        #     idx_ = np.argmax(avg_logit, axis=1)[0]

                        # idx_ = np.argmax(feat, axis=1)[0]

                        history_logit.append(feat)

                        history_logit = history_logit[-12:]

                        avg_logit = sum(history_logit)

                        idx_ = np.argmax(avg_logit, axis=1)[0]

                        idx = idx_

                        # idx, history = process_output(idx_, history)

                        t2 = time.time()
                        # print(f"{index} {catigories[idx]}")

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

    print(max_dict)
    print(min_dict)
    np.save('/home/zfy/tf_model/input/max_dict3.npy', max_dict)
    np.save('/home/zfy/tf_model/input/min_dict3.npy', min_dict)

    cap.release()
    cv2.destroyAllWindows()


main()
