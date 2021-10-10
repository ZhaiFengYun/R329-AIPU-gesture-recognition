#################
#@zhai fengyun -->Yanshan University
#################

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import torch
from tensorflow_core.python.framework import graph_util
from torch import nn

from mobilenet_v2_tsm import MobileNetV2

np.random.seed(1)
x = np.random.uniform(0, 1, [1, 3, 224, 224])

x0 = np.zeros([1, 3, 56, 56], dtype='float32')
x1 = np.zeros([1, 4, 28, 28], dtype='float32')
x2 = np.zeros([1, 4, 28, 28], dtype='float32')
x3 = np.zeros([1, 8, 14, 14], dtype='float32')
x4 = np.zeros([1, 8, 14, 14], dtype='float32')
x5 = np.zeros([1, 8, 14, 14], dtype='float32')
x6 = np.zeros([1, 12, 14, 14], dtype='float32')
x7 = np.zeros([1, 12, 14, 14], dtype='float32')
x8 = np.zeros([1, 20, 7, 7], dtype='float32')
x9 = np.zeros([1, 20, 7, 7], dtype='float32')
model_nodes = []
model_data_dict = np.load('/mnt/data/tf_tsm/tf_node_data.npy', allow_pickle=True)

quant_node=[]
def mobilenet_v2_arg_scope(weight_decay, is_training=True, depth_multiplier=1.0, regularize_depthwise=False,
                           dropout_keep_prob=1.0):
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'center': True, 'scale': True}):

        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=depth_multiplier):
                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob) as sc:
                    return sc


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def tf_conv_bn(net, oup, stride):
    net = tf.pad(net, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]])
    net = slim.conv2d(net, oup, [3, 3], stride=stride, padding='valid',
                      activation_fn=None, biases_initializer=None)
    quant_node.append(net.name)
    net=slim.batch_norm(net,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
    quant_node.append(net.name)



    # net = tf.layers.batch_normalization(
    #     net,
    #     axis=-1,
    #     momentum=0.9,
    #     epsilon=1e-5,
    #     center=True,
    #     scale=True,
    #     training=False)
    #model_nodes.append(net.name)

    net = tf.nn.relu6(net)
    quant_node.append(net.name)


    return net


def tf_conv_1x1_bn(net, oup):
    net = slim.conv2d(net, oup, [1, 1], stride=1, padding='valid',
                      activation_fn=None, biases_initializer=None)
    quant_node.append(net.name)


    # net = tf.layers.batch_normalization(
    #     net,
    #     axis=-1,
    #     momentum=0.9,
    #     epsilon=1e-5,
    #     center=True,
    #     scale=True,
    #     training=False)
    net = slim.batch_norm(net,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
    quant_node.append(net.name)



    net = tf.nn.relu6(net)
    quant_node.append(net.name)



    return net


def tf_make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def tf_InvertedResidual(net, inp, oup, stride, expand_ratio):
    hidden_dim = int(inp * expand_ratio)
    use_res_connect = stride == 1 and inp == oup

    if expand_ratio == 1:

        conv = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=[3, 3], stride=stride, padding='SAME',
                                     activation_fn=None, biases_initializer=None)

        quant_node.append(conv.name)


        conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
        quant_node.append(conv.name)



        # conv = tf.layers.batch_normalization(
        #     conv,
        #     axis=-1,
        #     momentum=0.9,
        #     epsilon=1e-5,
        #     center=True,
        #     scale=True,
        #     training=False)

        conv = tf.nn.relu6(conv)
        quant_node.append(conv.name)



        # print(conv.get_shape())

        conv = slim.conv2d(conv, oup, [1, 1], stride=1, padding='valid',
                           activation_fn=None, biases_initializer=None)
        quant_node.append(conv.name)
        # conv = tf.layers.batch_normalization(
        #     conv,
        #     axis=-1,
        #     momentum=0.9,
        #     epsilon=1e-5,
        #     center=True,
        #     scale=True,
        #     training=False)
        conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
        quant_node.append(conv.name)



    else:

        conv = slim.conv2d(net, hidden_dim, [1, 1], stride=1, padding='valid',
                           activation_fn=None, biases_initializer=None)
        quant_node.append(conv.name)
        # conv = tf.layers.batch_normalization(
        #     conv,
        #     axis=-1,
        #     momentum=0.9,
        #     epsilon=1e-5,
        #     center=True,
        #     scale=True,
        #     training=False)
        conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
        quant_node.append(conv.name)


        conv = tf.nn.relu6(conv)
        quant_node.append(conv.name)



        if stride == 1:
            conv = slim.separable_conv2d(inputs=conv, num_outputs=None, kernel_size=[3, 3], stride=stride,
                                         padding='SAME',
                                         activation_fn=None, biases_initializer=None)
            quant_node.append(conv.name)
            # conv = tf.layers.batch_normalization(
            #     conv,
            #     axis=-1,
            #     momentum=0.9,
            #     epsilon=1e-5,
            #     center=True,
            #     scale=True,
            #     training=False)
            conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
            quant_node.append(conv.name)

            conv = tf.nn.relu6(conv)
            quant_node.append(conv.name)



        else:
            conv = tf.pad(conv, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]])

            conv = slim.separable_conv2d(inputs=conv, num_outputs=None, kernel_size=[3, 3], stride=stride,
                                         padding='VALID',
                                         activation_fn=None, biases_initializer=None)
            quant_node.append(conv.name)
            # conv = tf.layers.batch_normalization(
            #     conv,
            #     axis=-1,
            #     momentum=0.9,
            #     epsilon=1e-5,
            #     center=True,
            #     scale=True,
            #     training=False)
            conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
            quant_node.append(conv.name)

            conv = tf.nn.relu6(conv)
            quant_node.append(conv.name)

        conv = slim.conv2d(conv, oup, [1, 1], stride=1, padding='valid',
                           activation_fn=None, biases_initializer=None)
        quant_node.append(conv.name)

        # conv = tf.layers.batch_normalization(
        #     conv,
        #     axis=-1,
        #     momentum=0.9,
        #     epsilon=1e-5,
        #     center=True,
        #     scale=True,
        #     training=False)
        conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
        quant_node.append(conv.name)




    if use_res_connect:
        return tf.add(net, conv)
    else:
        return conv


def tf_InvertedResidualWithShift(net, shift_buffer, inp, oup, stride, expand_ratio, n):
    assert stride in [1, 2]

    assert expand_ratio > 1

    hidden_dim = int(inp * expand_ratio)
    # print(stride)
    use_res_connect = stride == 1 and inp == oup
    # print(use_res_connect)
    assert use_res_connect

    b=net.shape[0]
    w = net.shape[1]
    h = net.shape[2]
    c=net.shape[3]


    net1 = tf.slice(net, [0, 0, 0, 0], [b, w, h, n])
    quant_node.append(net1.name)

    print('a',net1.shape)

    net2 = tf.slice(net, [0, 0, 0, n], [b, w, h, c-n])
    quant_node.append(net2.name)

    net3 = tf.concat([shift_buffer, net2], axis=3)
    quant_node.append(net3.name)

    conv = slim.conv2d(net3, hidden_dim, [1, 1], stride=1, padding='valid',
                       activation_fn=None, biases_initializer=None)
    quant_node.append(conv.name)
    # conv = tf.layers.batch_normalization(
    #     conv,
    #     axis=-1,
    #     momentum=0.9,
    #     epsilon=1e-5,
    #     center=True,
    #     scale=True,
    #     training=False)
    conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
    quant_node.append(conv.name)

    conv = tf.nn.relu6(conv)
    quant_node.append(conv.name)



    if stride == 1:
        conv = slim.separable_conv2d(inputs=conv, num_outputs=None, kernel_size=[3, 3], stride=stride,
                                     padding='SAME', activation_fn=None,
                                     biases_initializer=None)

        quant_node.append(conv.name)

        # conv = tf.layers.batch_normalization(
        #     conv,
        #     axis=-1,
        #     momentum=0.9,
        #     epsilon=1e-5,
        #     center=True,
        #     scale=True,
        #     training=False)
        conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
        quant_node.append(conv.name)



        conv = tf.nn.relu6(conv)
        quant_node.append(conv.name)


    else:
        conv = tf.pad(conv, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]])

        conv = slim.separable_conv2d(inputs=conv, num_outputs=None, kernel_size=[3, 3], stride=stride,
                                     padding='VALID',
                                     activation_fn=None, biases_initializer=None)
        quant_node.append(conv.name)
        # conv = tf.layers.batch_normalization(
        #     conv,
        #     axis=-1,
        #     momentum=0.9,
        #     epsilon=1e-5,
        #     center=True,
        #     scale=True,
        #     training=False)
        conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
        quant_node.append(conv.name)



        conv = tf.nn.relu6(conv)
        quant_node.append(conv.name)

    conv = slim.conv2d(conv, oup, [1, 1], stride=1, padding='valid',
                       activation_fn=None, biases_initializer=None)
    quant_node.append(conv.name)
    # conv = tf.layers.batch_normalization(
    #     conv,
    #     axis=-1,
    #     momentum=0.9,
    #     epsilon=1e-5,
    #     center=True,
    #     scale=True,
    #     training=False)
    conv = slim.batch_norm(conv,scale=True, decay=0.9,epsilon=1e-5,is_training=False)
    quant_node.append(conv.name)

    net4 = tf.add(net, conv)
    quant_node.append(net4.name)


    print('net4',net4)


    return net4, net1


def tf_MobileNetV2(net, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    n_class = 27
    input_size = 224
    width_mult = 1.

    input_channel = 32
    last_channel = 1280
    interverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    # building first layer
    assert input_size % 32 == 0
    input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
    last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
    features = tf_conv_bn(net, input_channel, 2)

    global_idx = 0
    shift_block_idx = [2, 4, 5, 7, 8, 9, 11, 12, 14, 15]

    shift_buffer = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    tensor_shape = [24, 32, 32, 64, 64, 64, 96, 96, 160, 160]

    shift_buffer_idx = 0

    return_buff=[]

    for t, c, n, s in interverted_residual_setting:
        output_channel = make_divisible(c * width_mult) if t > 1 else c
        for i in range(n):
            if i == 0:
                if global_idx in shift_block_idx:
                    features, buff = tf_InvertedResidualWithShift(features, shift_buffer[shift_buffer_idx],
                                                                  input_channel,
                                                                  output_channel, s, t, tensor_shape[shift_buffer_idx]//8)
                    return_buff.append(buff)
                    # print(buff.shape)

                    shift_buffer_idx += 1
                else:
                    features = tf_InvertedResidual(features, input_channel, output_channel, s, t)
                global_idx += 1

            else:

                if global_idx in shift_block_idx:
                    features, buff = tf_InvertedResidualWithShift(features, shift_buffer[shift_buffer_idx],
                                                                  input_channel,
                                                                  output_channel, 1, t, tensor_shape[shift_buffer_idx]//8)
                    return_buff.append(buff)
                    # print(buff.shape)
                    shift_buffer_idx += 1

                else:

                    features = tf_InvertedResidual(features, input_channel, output_channel, 1, t)

                global_idx += 1
            input_channel = output_channel

    features = tf_conv_1x1_bn(features, last_channel)
    #
    features = tf.reduce_mean(features, axis=2)
    features = tf.reduce_mean(features, axis=1)
    #
    features = slim.fully_connected(features, n_class, activation_fn=None, biases_initializer=slim.init_ops.zeros_initializer())
    quant_node.append(features.name)

    print(features)
    print('#######################')


    #return features,return_buff[0],return_buff[1],return_buff[2],return_buff[3],return_buff[4],return_buff[5],return_buff[6],return_buff[7],return_buff[8],return_buff[9]
    return features



if __name__ == '__main__':
    torch_module = MobileNetV2(n_class=27)

    shift_buffer = [torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7])]

    torch_module.eval()
    torch_module.load_state_dict(torch.load("/mnt/data/tsm-cannymotion2/online_demo/mobilenetv2_jester_online.pth.tar"))

    #print(torch_module)

    y0, buffer = torch_module(torch.Tensor(x), shift_buffer)
    #y0, buffer = torch_module(torch.Tensor(x), buffer)

    # #
    f = tf.convert_to_tensor(x.transpose([0, 2, 3, 1]))
    f = tf.cast(f, tf.float32)

    x0 = tf.convert_to_tensor(x0.transpose([0, 2, 3, 1]))
    x0 = tf.cast(x0, tf.float32)

    x1 = tf.convert_to_tensor(x1.transpose([0, 2, 3, 1]))
    x1 = tf.cast(x1, tf.float32)

    x2 = tf.convert_to_tensor(x2.transpose([0, 2, 3, 1]))
    x2 = tf.cast(x2, tf.float32)

    x3 = tf.convert_to_tensor(x3.transpose([0, 2, 3, 1]))
    x3 = tf.cast(x3, tf.float32)

    x4 = tf.convert_to_tensor(x4.transpose([0, 2, 3, 1]))
    x4 = tf.cast(x4, tf.float32)

    x5 = tf.convert_to_tensor(x5.transpose([0, 2, 3, 1]))
    x5 = tf.cast(x5, tf.float32)

    x6 = tf.convert_to_tensor(x6.transpose([0, 2, 3, 1]))
    x6 = tf.cast(x6, tf.float32)

    x7 = tf.convert_to_tensor(x7.transpose([0, 2, 3, 1]))
    x7 = tf.cast(x7, tf.float32)

    x8 = tf.convert_to_tensor(x8.transpose([0, 2, 3, 1]))
    x8 = tf.cast(x8, tf.float32)

    x9 = tf.convert_to_tensor(x9.transpose([0, 2, 3, 1]))
    x9 = tf.cast(x9, tf.float32)

    # y = tf_MobileNetV2(f, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    # #y, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = tf_MobileNetV2(f, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    # saver = tf.train.Saver()
    #
    # with tf.Session() as sess:
    #         #sess.run(tf.global_variables_initializer())
    #     # with slim.arg_scope([tf.layers.batch_normalization], is_training=True):
    #         saver.restore(sess, '/mnt/data/tf_tsm/new_ckpt2/model_ckpt')  # 加载到当前环境中
    #
    #         y = sess.run(y)
    #         print(y)
            #print(y.transpose([0, 3,1, 2]))
            #saver.save(sess, "/mnt/data/tf_tsm/ckpt2/model_ckpy")
            # x9=sess.run(x9)
            #
            # print((x9.transpose([0, 3,1, 2])-buffer[9].detach().numpy()).sum())
    #         op = sess.graph.get_operations()
    #
    #         for m in op:
    #             print(m.values())
    #
    #         print(y)


    # test_writer.add_summary(summary, j)
    # print('test_acc:' + str(np.mean(test_accuracy_list)))
    # print('test_loss:' + str(np.mean(test_loss_list)))



    xp = tf.placeholder(tf.float32, [1, 224, 224, 3], name='x_input')

    x0 = tf.placeholder(tf.float32, [1, 56, 56, 3], name='x_input0')
    x1 = tf.placeholder(tf.float32, [1, 28, 28, 4], name='x_input1')
    x2 = tf.placeholder(tf.float32, [1, 28, 28, 4], name='x_input2')
    x3 = tf.placeholder(tf.float32, [1, 14, 14, 8], name='x_input3')
    x4 = tf.placeholder(tf.float32, [1, 14, 14, 8], name='x_input4')
    x5 = tf.placeholder(tf.float32, [1, 14, 14, 8], name='x_input5')
    x6 = tf.placeholder(tf.float32, [1, 14, 14, 12], name='x_input6')
    x7 = tf.placeholder(tf.float32, [1, 14, 14, 12], name='x_input7')
    x8 = tf.placeholder(tf.float32, [1, 7, 7, 20], name='x_input8')
    x9 = tf.placeholder(tf.float32, [1, 7, 7, 20], name='x_input9')

    y = tf_MobileNetV2(xp, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    saver = tf.train.Saver()

    x0a = np.zeros([1, 56, 56, 3], dtype='float32')
    x1a = np.zeros([1, 28, 28, 4], dtype='float32')
    x2a = np.zeros([1, 28, 28, 4], dtype='float32')
    x3a = np.zeros([1, 14, 14, 8], dtype='float32')
    x4a = np.zeros([1, 14, 14, 8], dtype='float32')
    x5a = np.zeros([1, 14, 14, 8], dtype='float32')
    x6a = np.zeros([1, 14, 14, 12], dtype='float32')
    x7a = np.zeros([1, 14, 14, 12], dtype='float32')
    x8a = np.zeros([1, 7, 7, 20], dtype='float32')
    x9a = np.zeros([1, 7, 7, 20], dtype='float32')
    with tf.Session() as sess:

        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/mnt/data/tf_tsm/new_ckpt2/model_ckpt')
        output_node_names=['fully_connected/BiasAdd','Slice','Slice_2','Slice_4','Slice_6','Slice_8','Slice_10','Slice_12','Slice_14','Slice_16','Slice_18']
        #c_grph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['fully_connected/BiasAdd'])
    #     input_graph_def = tf.get_default_graph().as_graph_def()
    #     output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
    #                                                                     output_node_names)
    #
    #     #c_grph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['fully_connected/BiasAdd'])
    #
    #
    #
        print(quant_node)

        # with open("node.txt", "w") as f:
        #     f.write(str(quant_node))  #

        #np.savetxt('node.txt',str(quant_node))

        # for i in quant_node:
        #     print(i)
        for i in range(1):
            f = tf.convert_to_tensor(np.random.uniform(0, 1, [5, 224, 224, 3]))
            f = tf.cast(f, tf.float32)

            yl = tf.convert_to_tensor(np.zeros((5, 51)))
            yl = tf.cast(yl, tf.int16)

            f = x.transpose([0, 2, 3, 1])



            feed_dict={xp: f,
                            x0: x0a,
                            x1: x1a,
                            x2: x2a,
                            x3: x3a,
                            x4: x4a,
                            x5: x5a,
                            x6: x6a,
                            x7: x7a,
                            x8: x8a,
                            x9: x9a
                              }
            #y_p,x0a,x1a,x2a,x3a,x4a,x5a, x6a,x7a,x8a,x9a=sess.run(y,feed_dict)
            y_p = sess.run(y, feed_dict)
            sess.run()

            print(y_p)
    #
    # with tf.gfile.FastGFile('/mnt/data/tf_tsm/' + 'model3.pb', mode='wb') as f:
    #     f.write(output_graph_def.SerializeToString())



 



