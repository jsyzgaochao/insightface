
import sys
import os
import mxnet as mx
# import symbol_octconv as sym
import mxnet.symbol as sym
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config


def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    if act_type=='prelu':
      body = sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = sym.Activation(data=data, act_type=act_type, name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' %(name, suffix))
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)    
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv

def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj

def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity

def SEModule(data, num_filter, name):
    body = sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
    body = sym.Convolution(data=body, num_filter=num_filter//8, kernel=(1,1), stride=(1,1), pad=(0,0),
                              name=name+"_se_conv1")
    body = Act(data=body, act_type="prelu", name=name+'_se_relu1')
    body = sym.Convolution(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                              name=name+"_se_conv2")
    body = sym.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
    data = sym.broadcast_mul(data, body)
    return data

def get_symbol():
    num_classes = config.emb_size
    print('in_network', config)
    fc_type = config.net_output
    data = sym.Variable(name="data")
    data = data-127.5
    data = data*0.0078125
    blocks = config.net_blocks
    conv_1 = Conv(data, num_filter=48, kernel=(5, 5), pad=(2, 2), stride=(2, 2), name="conv_1")
    if blocks[0]==1:
      conv_2_dw = Conv(conv_1, num_group=48, num_filter=48, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw")
    else:
      conv_2_dw = Residual(conv_1, num_block=blocks[0], num_out=48, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=48, name="res_2")

    conv_23 = DResidual(conv_2_dw, num_out=64, kernel=(5, 5), stride=(2, 2), pad=(2, 2), num_group=128, name="dconv_23")
    conv_3a = Residual(conv_23, num_block=blocks[1]//2, num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, name="res_3a")
    conv_3ase = SEModule(conv_3a, 64, "res_3ase")
    conv_3b = Residual(conv_3ase, num_block=blocks[1]-blocks[1]//2, num_out=64, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_group=128, name="res_3b")
    conv_3bse = SEModule(conv_3b, 64, "res_3bse")

    conv_34 = DResidual(conv_3bse, num_out=128, kernel=(5, 5), stride=(2, 2), pad=(2, 2), num_group=256, name="dconv_34")
    conv_4a = Residual(conv_34, num_block=blocks[2]//2, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_4a")
    conv_4ase = SEModule(conv_4a, 128, "res_4ase")
    conv_4b = Residual(conv_4ase, num_block=blocks[2]-blocks[2]//2, num_out=128, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_group=256, name="res_4b")
    conv_4bse = SEModule(conv_4b, 128, "res_4bse")

    conv_45 = DResidual(conv_4bse, num_out=160, kernel=(5, 5), stride=(2, 2), pad=(2, 2), num_group=512, name="dconv_45")
    conv_5a = Residual(conv_45, num_block=blocks[3]//2, num_out=160, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=480, name="res_5a")
    conv_5ase = SEModule(conv_5a, 160, "res_5ase")
    conv_5b = Residual(conv_5ase, num_block=blocks[3]-blocks[3]//2, num_out=160, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_group=480, name="res_5b")
    conv_5bse = SEModule(conv_5b, 160, "res_5bse")

    conv_6_sep = Conv(conv_5bse, num_filter=640, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")

    fc1 = symbol_utils.get_fc1(conv_6_sep, num_classes, fc_type, input_channel=640)
    return fc1
