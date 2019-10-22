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

def yolo_head(features,anchors,num_classes,input_shape,calc_loss=False):
    """
    解析yolo网络层的输出，转换为 bounding box
    """
    num_anchors = len(anchors)
    # reshape to batch,height,width,num_anchors,box_params
    anchors_tensor = tf.reshape(tf.constant(anchors),[1,1,1,num_anchors,2])

    grid_shape = features.shape[1:3] # grid cell => width x height
    grid_y = tf.tile(tf.reshape(tf.range(0,grid_shape[0]),[-1,1,1,1]),[1,grid_shape[1],1,1])
    grid_x = tf.tile(tf.reshape(tf.range(0,grid_shape[1]),[1,-1,1,1]),[grid_shape[0],1,1,1])
    grid = tf.concat([grid_x,grid_y],axis=-1)
    grid = tf.cast(grid,features.dtype)

    # width x height x num_anchors x (num_classes + 5)
    features = tf.reshape(features,[grid_shape[0],grid_shape[1],num_anchors,num_classes + 5])

    # 根据预测值所在的grid cell位置和起对应的anchor box，调整输出
    box_xy = (tf.math.sigmoid(features[...,:2]) + grid) / tf.cast(grid_shape[::-1],features.dtype)
    box_wh = tf.math.exp(features[...,2:4]) * anchors_tensor / tf.cast(input_shape[::-1],features.dtype)

    box_confidence = tf.math.sigmoid(features[...,4:5]) 
    box_class_probs = tf.math.sigmoid(features[...,5])

    if calc_loss == True:
        return grid,features,box_xy,box_wh

    return box_xy,box_wh,box_confidence,box_class_probs

def yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape):

    box_yx = box_xy[...,::-1]
    box_hw = box_wh[...,::-1]
    input_shape = tf.cast(input_shape,tf.float32)
    image_shape = tf.cast(image_shape,tf.float32)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx - (box_hw / 2.0)

    boxes = tf.concat([
        box_mins[...,0:1],
        box_mins[...,1:2],
        box_maxes[...,0:1],
        box_maxes[...,1:2]
    ])

    boxes *= tf.concat([image_shape,image_shape])
    retur boxes

def yolo_boxes_and_scores(features,anchors,num_classes,input_shape,image_shape):
    box_xy,box_wh,box_confidence,box_class_probs = yolo_head(features,anchors,num_classes,input_shape)

    boxes = yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape)
    boxes = tf.reshape(boxes,[-1,4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores,[-1,num_classes])
    return boxes,box_scores

def yolo_eval(yolo_outputs,anchors,num_classes,image_shape,max_boxes=20,score_threshold=0.6,iou_threshold=0.5):
    num_layers = len(yolo_outputs)

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = yolo_outputs[0].shape[1:3] * 32

    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes,_box_scores = yolo_boxes_and_scores(yolo_outputs[l],anchores[anchor_mask[l]],num_classes,input_shape,image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = tf.concat(boxes,axis=0)
    box_scores = tf.concat(box_scores,axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(max_boxes,dtype=tf.int32)
    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes,mask[:,c])
        class_box_scores = tf.boolean_mask(box_scores[:,c],mask[:,c])
        
        nms_index = tf.image.non_max_suppression(
            class_boxes,class_box_scores,max_boxes_tensor,iou_threshold=iou_threshold
        )
        class_boxes = tf.gather(class_boxes,nms_index)
        class_box_scores = tf.gather(class_box_scores,nms_index)
        classes = tf.ones_like(class_box_scores,tf.int32) * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = tf.concat(boxes_,axis=0)
    scores_ = tf.concat(scores_,axis=0)
    classes_ = tf.concat(classes_,axis=0)

    return boxes_,scores_,classes_

def box_iou(b1,b2):
    
    b1 = tf.expand_dims(b1,-2)
    b1_xy = b1[...,:2]
    b1_wh = b1[...,2:4]
    b1_wh_half = b1_wh / 2.0
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2 = tf.exp(b2,0)
    b2_xy = [...,:2]
    b2_wh = [...,2:4]
    b2_wh_half = b2_wh / 2.0
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins,b2_mins)
    intersect_maxes = tf.minimum(b1_maxes,b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins,0.)
    intersect_areas = intersect_wh[...,0] * intersect_wh[...,1]

    b1_area = b1_wh[...,0] * b1_wh[...,1]
    b2_area = b2_wh[...,0] * b2_wh[...,1]

    iou = intersect_areas / (b1_area + b2_area - intersect_areas)

    return iou

def yolo_loss(args,anchors,num_classes,ignore_thresh = 0.5,print_loss = False):
    
    num_layers = len(anchors) // 3 
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8],[3,4,5],[0,1,2]] if num_layers == 3 else [[3,4,5],[0,1,2]]
    input_shape = tf.cast(yolo_outputs[0].shape[1:3] * 32,y_true[0].dtype)
    grid_shapes = [tf.cast(yolo_outputs[l])[1:3],dtype=y_true[0].dtype) for l in range(num_layers)]
    loss = 0
    m = yolo_outputs[0].shape[0]
    mf = tf.cast(m,tf.float32)

    for l in range(num_layers):
        object_mask = y_true[l][...,4:5] # object mask
        true_class_probs = y_true[l][...,5]

        grid,raw_pred,pred_xy,pred_wh = yolo_head(yolo_ouputs[l],anchors[anchor_mask[l]],num_classes,input_shape,calc_loss=True)

        pred_box = tf.concat([pred_xy,pred_wh])

        raw_true_xy = y_true[l][...,:2] * grid_shapes[l][::-1] - grid 
        raw_true_wh = y_true[l][...,2:4] / anchors[anchor_mask[l]] * input_shape[::-1]
        raw_true_wh = tf.switch(object_mask,raw_true_wh,tf.zeros_like(raw_true_wh))
        box_loss_scale = 2 - y_true[l][...,2:3] * y_true[l][...,3:4]

        ignore_mask = tf.TensorArray(y_true[0].dtype,size=1,dynamic_size=True)
        object_mask_bool = tf.cast(object_mask,dtype=tf.bool)

        def loop_body(b,ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4],object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b],true_box)
            best_iou = tf.reduce_max(iou,axis=-1)
            ignore_mask = ignore_mask.write(b,tf.cast(best_iou < ignore_thresh,true_box.dtype))
            return b + 1,ignore_mask
        _,ignore_mask = tf.control_flow_ops.while_loop(lambda b,*args : b < m,loop_body,[0,ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask,-1)

        xy_loss = object_mask * box_loss_scale * tf.binary_crossentropy(raw_true_xy,raw_pred[...,0:2],from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[...,2:4])
        confidence_loss = object_mask * tf.binary_crossentropy(object_mask,raw_pred[...,4:5],from_logits=True) + \
            (1 - object_mask) * tf.binary_crossentropy(object_mask,raw_pred[...,4:5],from_logits=True) * ignore_mask
        class_loss = object_mask * tf.binary_crossentropy(true_class_probs,raw_pred[...,5:],from_logits=True)


        xy_loss = tf.reduce_sum(xy_loss) / mf 
        wh_loss = tf.reduce_sum(wh_loss) / mf 
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')

    return loss



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