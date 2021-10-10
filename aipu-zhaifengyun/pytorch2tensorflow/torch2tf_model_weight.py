
#################
#@zhai fengyun -->Yanshan University
#################


# model_node = ['Conv/Relu6:0', 'SeparableConv2d/Relu6:0', 'Conv_1/Conv2D:0', 'Conv_2/Relu6:0',
#                   'SeparableConv2d_1/Relu6:0', 'Conv_3/Conv2D:0', 'Conv_4/Relu6:0', 'SeparableConv2d_2/Relu6:0',
#                   'Conv_5/Conv2D:0', 'Conv_6/Relu6:0', 'SeparableConv2d_3/Relu6:0', 'Conv_7/Conv2D:0', 'Conv_8/Relu6:0',
#                   'SeparableConv2d_4/Relu6:0', 'Conv_9/Conv2D:0', 'Conv_10/Relu6:0', 'SeparableConv2d_5/Relu6:0',
#                   'Conv_11/Conv2D:0', 'Conv_12/Relu6:0', 'SeparableConv2d_6/Relu6:0', 'Conv_13/Conv2D:0',
#                   'Conv_14/Relu6:0', 'SeparableConv2d_7/Relu6:0', 'Conv_15/Conv2D:0', 'Conv_16/Relu6:0',
#                   'SeparableConv2d_8/Relu6:0', 'Conv_17/Conv2D:0', 'Conv_18/Relu6:0', 'SeparableConv2d_9/Relu6:0',
#                   'Conv_19/Conv2D:0', 'Conv_20/Relu6:0', 'SeparableConv2d_10/Relu6:0', 'Conv_21/Conv2D:0',
#                   'Conv_22/Relu6:0', 'SeparableConv2d_11/Relu6:0', 'Conv_23/Conv2D:0', 'Conv_24/Relu6:0',
#                   'SeparableConv2d_12/Relu6:0', 'Conv_25/Conv2D:0', 'Conv_26/Relu6:0', 'SeparableConv2d_13/Relu6:0',
#                   'Conv_27/Conv2D:0', 'Conv_28/Relu6:0', 'SeparableConv2d_14/Relu6:0', 'Conv_29/Conv2D:0',
#                   'Conv_30/Relu6:0', 'SeparableConv2d_15/Relu6:0', 'Conv_31/Conv2D:0', 'Conv_32/Relu6:0',
#                   'SeparableConv2d_16/Relu6:0', 'Conv_33/Conv2D:0', 'Conv_34/Relu6:0', 'fully_connected/MatMul:0']
#
#
#
#
#
def split_str(str):
    i = 0
    new_str = ''
    while str[i] != '/':

        new_str = new_str + str[i]
        i = i + 1

    return new_str
#
# model_new_node=[]
#
# for i in range(len(model_node)):
#     model_new_node.append(split_str(model_node[i]))
#     #print(model_node[i])
#
# print(model_new_node)


import tensorflow as tf
import argparse
import os
import numpy as np

model_data_dict = np.load('/mnt/data/tf_tsm/tf_node_data.npy', allow_pickle=True)

for k in model_data_dict.item():
    print(k,model_data_dict.item()[k].shape)



parser = argparse.ArgumentParser(description='')

parser.add_argument("--checkpoint_path", default='/mnt/data/tf_tsm/ckpt2', help="restore ckpt")  # 原参数路径
parser.add_argument("--new_checkpoint_path", default='../deeplab_resnet_altered/', help="path_for_new ckpt")  # 新参数保存路径
parser.add_argument("--add_prefix", default='deeplab_v2/', help="prefix for addition")  # 新参数名称中加入的前缀名

args = parser.parse_args()

# batch_normalization/beta
# batch_normalization/gamma
# batch_normalization/moving_mean
# batch_normalization/moving_variance
# batch_normalization_1/beta
# batch_normalization_1/gamma
# batch_normalization_1/moving_mean
# batch_normalization_1/moving_variance


def main():
    if not os.path.exists(args.new_checkpoint_path):
        os.makedirs(args.new_checkpoint_path)
    with tf.Session() as sess:


        new_var_list=[]
        for var_name, _ in tf.contrib.framework.list_variables(args.checkpoint_path):  # 得到checkpoint文件中所有的参数（名字，形状）元组

            var = tf.contrib.framework.load_variable(args.checkpoint_path, var_name)  # 得到上述参数的值






            w = model_data_dict.item()[var_name]



            if 'fully_connected/weights' in var_name:
                w = w.transpose([1, 0])
            elif '/weights' in var_name:

                w = w.transpose([2, 3, 1, 0])
            elif 'depthwise_weights' in var_name:

                w = w.transpose([2, 3, 0, 1])
            else:
                w=w

            print(var_name, var.shape, w.shape)



            #print(var.shape,w.shape)



            reset_var = tf.Variable(w, name=var_name)
            new_var_list.append(reset_var)  # 把赋予新名称的参数加入空列表

        chekt_path='/mnt/data/tf_tsm/new_ckpt2/model_ckpt'
        sess.run(tf.global_variables_initializer())  # 初始化一下参数（这一步必做）

        saver = tf.train.Saver(var_list=new_var_list)  # 构造一个保存器


        saver.save(sess, chekt_path)  # 直接进行保存

        print('ok')






if __name__ == '__main__':
    main()
