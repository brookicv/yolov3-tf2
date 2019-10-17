from functools import wraps

import tensorflow as tf
import tensorflow.keras.layers import Conv2D,Add,ZeroPadding2D,UpSampling2D,Concatenate,MaxPooling2D
import tensorflow.keras.layers import LeakyReLU,BatchNormalization,Input 
import tensorflow.keras.models import Model 
import tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.regularizers import l2

from yolo3.utils import compose 

@wraps(Conv2D)
def DarkNetConv2D(*args,**kwargs):
    darknet_conv_kwargs = {"kernel_regularizer":l2(5e-4)}
    darknet_conv_kwargs["padding"] = "valid" if kwargs.get("strides") == (2,2) else "same"
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args,**darknet_conv_kwargs)

def DarkNetConv2D_BN_Leaky(*args,**kwargs):
    """ conv2d + BatchNormalization + LeakyReLU""" 
    no_bias_kwargs = {"use_bias":False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarkNetConv2D(*args,**no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
    )

def resblock_body(x,num_filters,num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarkNetConv2D_BN_Leaky(num_filters,(3,3),strides=(2,2))(x) # down sampling
    for i in range(num_blocks):
        y = compose(
            DarkNetConv2D_BN_Leaky(num_filters // 2 , (1,1)), # 1 x 1 conv2d
            DarkNetConv2D_BN_Leaky(num_filters,(3,3))
        )(x)
        x = Add()([x,y])

    return x

def make_last_layers(x,num_filters,out_filters):
    x = compose(
        DarkNetConv2D_BN_Leaky(num_filters,(1,1)), # 1 x 1 conv2d
        DarkNetConv2D_BN_Leaky(num_filters * 2,(3,3)), # 3 x 3 conv2d
        DarkNetConv2D_BN_Leaky(num_filters,(1,1)), # 1 x 1 conv2d
        DarkNetConv2D_BN_Leaky(num_filters * 2,(3,3)),
        DarkNetConv2D_BN_Leaky(num_filters,(1,1)), # 1 x 1 conv2d
    )(x)

    y = compose(
        DarkNetConv2D_BN_Leaky(num_filters * 2,(3,3)),
        DarkNetConv2D_BN_Leaky(out_filters,(1,1))
    )(x)

    return x,y

def darkent_body(x):
    x = DarkNetConv2D_BN_Leaky(32,(3,3))(x)
    x = resblock_body(x,64,1)
    x = resblock_body(x,128,2)
    x = resblock_body(x,256,8)
    x = resblock_body(x,512,8)
    x = resblock_body(x,1024,4)

    return x  

def yolo_body(inputs,num_anchors,num_classes):
    darknet = Model(inputs,darkent_body(inputs))

    # 3 branches

    # 1 branch -> 13 x 13
    x ,y1 = make_last_layers(darknet.output,512,num_anchors*(num_classes + 5)) 

    # input -> 2 branch
    x = compose(
        DarkNetConv2D_BN_Leaky(256,(1,1)), # 1 x 1 conv2d
        UpSampling2D(2)
    )(x)
    x = Concatenate()([x,darknet.layers[152].output])
    # 2 branch -> 26 x 26
    x ,y2 = make_last_layers(x,256,num_anchors * (num_classes + 5))

    # input -> 3 branch
    x = compose(
        DarkNetConv2D_BN_Leaky(128,(1,1)), # 1 x 1 conv2d
        UpSampling2D(2)
    )(x)
    x = Concatenate()([x,darknet.layers[92].output])
    # 3 branch -> 52 x 52
    x,y3 = make_last_layers(x,128,num_anchors * (num_classes + 5))

    # yolov3-model
    return Model(inputs,[y1,y2,y3])