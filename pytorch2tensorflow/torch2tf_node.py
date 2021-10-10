#################
#@zhai fengyun -->Yanshan University
#################

import numpy
import torch.nn as nn
import torch
import math

from onnx_tf.backend import prepare


def conv_bn(inp, oup, stride):
    print('strid', stride)
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        assert stride in [1, 2]

        self.a, self.b, self.c, self.d = inp, oup, stride, expand_ratio

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        # print('###############')
        #
        #
        #
        # print('inp:',self.a,'oup:',self.b, 'strid:', self.c, 'expand_ratio',self.d)
        #
        # print(x.shape)

        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualWithShift(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualWithShift, self).__init__()
        self.stride = stride

        assert stride in [1, 2]

        assert expand_ratio > 1

        hidden_dim = int(inp * expand_ratio)

        print('asdsad:', stride)
        self.use_res_connect = self.stride == 1 and inp == oup
        assert self.use_res_connect

        self.a, self.b, self.c, self.d = inp, oup, stride, expand_ratio

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x, shift_buffer, c):
        # print('shape:',x.size(1))

        # print('inp:',self.a,'oup:',self.b, 'strid:', self.c, 'expand_ratio',self.d)
        # print(x.shape,shift_buffer.shape)
        # c = x.size(1)
        # print(c)
        # print('size:',x.size())

        x1, x2 = x[:, : c // 8], x[:, c // 8:]
        return x + self.conv(torch.cat((shift_buffer, x2), dim=1)), x1


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
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
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        global_idx = 0
        shift_block_idx = [2, 4, 5, 7, 8, 9, 11, 12, 14, 15]
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c

            print(output_channel)
            for i in range(n):
                if i == 0:
                    block = InvertedResidualWithShift if global_idx in shift_block_idx else InvertedResidual

                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    global_idx += 1
                else:

                    block = InvertedResidualWithShift if global_idx in shift_block_idx else InvertedResidual

                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    global_idx += 1
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.ModuleList(self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()
        self.tensor_shape = [24, 32, 32, 64, 64, 64, 96, 96, 160, 160]

    def forward(self, x, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):

        shift_buffer = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]

        shift_buffer_idx = 0
        out_buffer = []

        for f in self.features:
            if isinstance(f, InvertedResidualWithShift):
                x, s = f(x, shift_buffer[shift_buffer_idx], self.tensor_shape[shift_buffer_idx])
                shift_buffer_idx += 1
                out_buffer.append(s)
            else:

                x = f(x)
        print('qaz:', x.shape)
        x = x.mean(3).mean(2)
        print(x.shape)
        x = self.classifier(x)

        # return x, out_buffer[0], out_buffer[1], out_buffer[2], out_buffer[3], out_buffer[4], out_buffer[5], out_buffer[
        #     6], out_buffer[7], out_buffer[8], out_buffer[9]

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2_140():
    return MobileNetV2(width_mult=1.4)


if __name__ == '__main__':
    import onnxruntime as onnrun

    import onnx



    print(MobileNetV2(27))

    # model_node = ['Conv/Relu6:0', 'SeparableConv2d/Relu6:0', 'Conv_1/Conv2D:0', 'Conv_2/Relu6:0',
    #               'SeparableConv2d_1/Relu6:0', 'Conv_3/Conv2D:0', 'Conv_4/Relu6:0', 'SeparableConv2d_2/Relu6:0',
    #               'Conv_5/Conv2D:0', 'Conv_6/Relu6:0', 'SeparableConv2d_3/Relu6:0', 'Conv_7/Conv2D:0', 'Conv_8/Relu6:0',
    #               'SeparableConv2d_4/Relu6:0', 'Conv_9/Conv2D:0', 'Conv_10/Relu6:0', 'SeparableConv2d_5/Relu6:0',
    #               'Conv_11/Conv2D:0', 'Conv_12/Relu6:0', 'SeparableConv2d_6/Relu6:0', 'Conv_13/Conv2D:0',
    #               'Conv_14/Relu6:0', 'SeparableConv2d_7/Relu6:0', 'Conv_15/Conv2D:0', 'Conv_16/Relu6:0',
    #               'SeparableConv2d_8/Relu6:0', 'Conv_17/Conv2D:0', 'Conv_18/Relu6:0', 'SeparableConv2d_9/Relu6:0',
    #               'Conv_19/Conv2D:0', 'Conv_20/Relu6:0', 'SeparableConv2d_10/Relu6:0', 'Conv_21/Conv2D:0',
    #               'Conv_22/Relu6:0', 'SeparableConv2d_11/Relu6:0', 'Conv_23/Conv2D:0', 'Conv_24/Relu6:0',
    #               'SeparableConv2d_12/Relu6:0', 'Conv_25/Conv2D:0', 'Conv_26/Relu6:0', 'SeparableConv2d_13/Relu6:0',
    #               'Conv_27/Conv2D:0', 'Conv_28/Relu6:0', 'SeparableConv2d_14/Relu6:0', 'Conv_29/Conv2D:0',
    #               'Conv_30/Relu6:0', 'SeparableConv2d_15/Relu6:0', 'Conv_31/Conv2D:0', 'Conv_32/Relu6:0',
    #               'SeparableConv2d_16/Relu6:0', 'Conv_33/Conv2D:0', 'Conv_34/Relu6:0', 'fully_connected/MatMul:0']

    model_node = [ 'Conv_1', 'Conv_2', 'SeparableConv2d_1', 'Conv_3', 'Conv_4',
                  'SeparableConv2d_2',
                  'Conv_5', 'Conv_6', 'SeparableConv2d_3', 'Conv_7', 'Conv_8', 'SeparableConv2d_4', 'Conv_9', 'Conv_10',
                  'SeparableConv2d_5', 'Conv_11', 'Conv_12', 'SeparableConv2d_6', 'Conv_13', 'Conv_14',
                  'SeparableConv2d_7',
                  'Conv_15', 'Conv_16', 'SeparableConv2d_8', 'Conv_17', 'Conv_18', 'SeparableConv2d_9', 'Conv_19',
                  'Conv_20',
                  'SeparableConv2d_10', 'Conv_21', 'Conv_22', 'SeparableConv2d_11', 'Conv_23', 'Conv_24',
                  'SeparableConv2d_12',
                  'Conv_25', 'Conv_26', 'SeparableConv2d_13', 'Conv_27', 'Conv_28', 'SeparableConv2d_14', 'Conv_29',
                  'Conv_30',
                  'SeparableConv2d_15', 'Conv_31', 'Conv_32', 'SeparableConv2d_16', 'Conv_33', 'Conv_34',
                   ]

    mode_new_node = []
    ck = 1
    sk = 1
    mode_new_node.append('Conv/weights')
    mode_new_node.append('SeparableConv2d/depthwise_weights')
    for v in model_node:
        if v[0] == 'C':
            mode_new_node.append('Conv_' + str(ck) + '/weights')
            # print('Conv_'+str(ck)+'/weights')
            ck = ck + 1
        if v[0] == 'S':
            mode_new_node.append('SeparableConv2d_' + str(sk) + '/depthwise_weights')
            # print('SeparableConv2d_'+str(sk)+'/depthwise_weights')
            sk = sk + 1
    mode_new_node.append('fully_connected/weights')

    chept = torch.load("mobilenetv2_jester_online.pth.tar")
    # i=0
    #
    # g=[]
    #
    # Conv9/weights(1, 1, 192, 32)
    # SeparableConv2d/depthwise_weights(3, 3, 32, 1)
    # batch_normalization_17/beta(32, ) #bias
    # batch_normalization_17/gamma(32, )
    # batch_normalization_17/moving_mean(32, )
    # batch_normalization_17/moving_variance(32, )

    # batch_normalization_9 / beta(144, )
    # batch_normalization_9 / gamma(144, )
    # batch_normalization_9 / moving_mean(144, )
    # batch_normalization_9 / moving_variance(144, )
    # fully_connected/weights(1280, 27)

    #fully_connected/biases


    tf_node_data = dict()
    i = 0
    for k, v in chept.items():

        print(k)

        if 'bias' not in k and 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k:
            if len(v.shape) > 1:


                name = mode_new_node[i]

                tf_node_data[name] = v.numpy()
                i = i + 1
                print(k)

        if k=='classifier.bias':
            tf_node_data['fully_connected/biases']=v


                # print(k,v.shape)

    #numpy.save('/mnt/data/tf_tsm/tf_node_data.npy', tf_node_data)
    #print(tf_node_data)
    # features
    # .18
    # .1.weight
    # features
    # .18
    # .1.bias
    # features
    # .18
    # .1.running_mean
    # features
    # .18
    # .1.running_var
    i=0
    for k, v in chept.items():

        if 'running_var' in k:
            beta_name=k[:-11]+'bias'
            gamma_name=k[:-11]+'weight'

            mean_name=k[:-11]+'running_mean'

            beta=chept[beta_name]
            gamma=chept[gamma_name]
            mean=chept[mean_name]
            var=v
            # if i==0:
            #     tf_node_data['batch_normalization/beta'] = beta.numpy()
            #     tf_node_data['batch_normalization/gamma'] = gamma.numpy()
            #     tf_node_data['batch_normalization/moving_mean'] = mean.numpy()
            #     tf_node_data['batch_normalization/moving_variance'] = var.numpy()
            # else:
            #     tf_node_data['batch_normalization_' + str(i) + '/beta'] = beta.numpy()
            #     tf_node_data['batch_normalization_' + str(i) + '/gamma'] = gamma.numpy()
            #     tf_node_data['batch_normalization_' + str(i) + '/moving_mean'] = mean.numpy()
            #     tf_node_data['batch_normalization_' + str(i) + '/moving_variance'] = var.numpy()
            # i=i+1
            if i==0:
                tf_node_data['BatchNorm/beta'] = beta.numpy()
                tf_node_data['BatchNorm/gamma'] = gamma.numpy()
                tf_node_data['BatchNorm/moving_mean'] = mean.numpy()
                tf_node_data['BatchNorm/moving_variance'] = var.numpy()
            else:
                tf_node_data['BatchNorm_' + str(i) + '/beta'] = beta.numpy()
                tf_node_data['BatchNorm_' + str(i) + '/gamma'] = gamma.numpy()
                tf_node_data['BatchNorm_' + str(i) + '/moving_mean'] = mean.numpy()
                tf_node_data['BatchNorm_' + str(i) + '/moving_variance'] = var.numpy()
            i=i+1



    numpy.save('/mnt/data/tf_tsm/tf_node_data.npy', tf_node_data)
    print(tf_node_data)

    #BatchNorm_














