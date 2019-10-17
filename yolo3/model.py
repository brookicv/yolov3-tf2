from functools import wraps

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Add,ZeroPadding2D,UpSampling2D,Concatenate,MaxPooling2D
from tensorflow.keras.layers import LeakyReLU,BatchNormalization,Input 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.regularizers import l2

from utils import compose 

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
        DarkNetConv2D_BN_Leaky(num_filters,(1,1))# 1 x 1 conv2d
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


def tiny_yolo_body(inputs,num_anchors,num_classes):
    x1 = compose(
        DarkNetConv2D_BN_Leaky(16,(3,3)),
        MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same"),
        DarkNetConv2D_BN_Leaky(32,(3,3)),
        MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same"),
        DarkNetConv2D_BN_Leaky(64,(3,3)),
        MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same"),
        DarkNetConv2D_BN_Leaky(128,(3,3)),
        MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same"),
        DarkNetConv2D_BN_Leaky(256,(3,3))
    )(inputs)

    x2 = compose(
        MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same"),
        DarkNetConv2D_BN_Leaky(512,(3,3)),
        MaxPooling2D(pool_size=(2,2),strides=(1,1),padding="same"),
        DarkNetConv2D_BN_Leaky(1024,(3,3)),
        DarkNetConv2D_BN_Leaky(256,(1,1))
    )(x1)

    # output layer
    y1 = compose(
        DarkNetConv2D_BN_Leaky(512,(3,3)),
        DarkNetConv2D(num_anchors * (num_classes + 5),(1,1))
    )(x2)

    x2 = compose(
        DarkNetConv2D_BN_Leaky(128,(1,1)),
        UpSampling2D(2)
    )(x2)
    y2 = compose(
        Concatenate(),
        DarkNetConv2D_BN_Leaky(256,(3,3)),
        DarkNetConv2D(num_anchors * (num_classes + 5),(1,1))
    )([x2,x1])

    return Model(inputs,[y1,y2])

def mobile_yolo_body(inputs,num_anchors,num_classes):
    mobile_net = MobileNet(input_tensor=inputs,weights="imagenet")

    # conv_pw_13_relu: 13 x 13 x 1024
    # conv_pw_11_relu: 26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256

    f1 = mobile_net.get_layer("conv_pw_13_relu").output
    x,y1 = make_last_layers(f1,512,num_anchors*(num_classes + 5))

    x = compose(
        DarkNetConv2D_BN_Leaky(256,(1,1)),
        UpSampling2D(2)
    )(x)
    f2 = mobile_net.get_layer("conv_pw_11_relu").output
    x = Concatenate()([x,f2])

    x,y2 = make_last_layers(x,256,num_anchors * (num_classes + 5))

    x = compose(DarkNetConv2D_BN_Leaky(128,(1,1)),UpSampling2D(2))(x)
    f3 = mobile_net.get_layer("conv_pw_5_relu").output

    x = Concatenate()([x,f3])
    x,y3 = make_last_layers(x,128,num_anchors * (num_classes + 5))

    return Model(inputs=inputs,outputs=[y1,y2,y3])

if __name__ == "__main__":
    image_input = Input(shape=(416,416,3))

    yolo = yolo_body(image_input,3,1)
    # yolo.summary()

    print(yolo.output[0].shape)
    print(yolo.output[1].shape)
    print(yolo.output[2].shape)

    tiny_yolo = tiny_yolo_body(image_input,3,1)
    # tiny_yolo.summary()

    print(tiny_yolo.output[0].shape)
    print(tiny_yolo.output[1].shape)

    mobile_yolo = mobile_yolo_body(image_input,3,1)
    mobile_yolo.summary()
    print(mobile_yolo.output[0].shape)
    print(mobile_yolo.output[1].shape)
    print(mobile_yolo.output[2].shape)