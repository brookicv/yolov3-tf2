from functools import wraps

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Add,ZeroPadding2D,UpSampling2D,Concatenate,MaxPooling2D
from tensorflow.keras.layers import LeakyReLU,BatchNormalization,Input 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.regularizers import l2

from PIL import Image

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

def preprocess_true_boxes(true_boxes,input_shape,anchors,num_classes):
    """
    Preprocess true boxes tor training input format
    
    Parameters:
        true_boxes: shape=(b,T,5) . b,batch size; T, maximum boxes in a traing image;
        input_shape: (h,w) multiples of 32
        anchors : array,shape=(N,2),(w,w)
        num_classes : count of classes
    """
    
    num_layers = len(anchors) // 3 # yolo 网络层输出层的个数，默认为3. 13x13,26x26,52x52
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # tiny-yolo 的网络层输出只有2个
    
    true_boxes = np.array(true_boxes,dtype=np.float32)
    input_shape = np.array(input_shape,dtype=np.int32)

    boxes_xy = (true_boxes[...,0:2] + true_boxes[...,2:4]) // 2 # bbox的中心点坐标
    boxes_wh = (true_boxes[...,2:4] - true_boxes[...,0:2]) # bbox的宽和高

    # 将bbox(x,y,w,h)对416x416做归一化
    true_boxes[...,0:2] = true_boxes[...,0:2] / input_shape[::-1]
    true_boxes[...,2:4] = true_boxes[...,2:4] / input_shape[::-1]

    m = true_boxes.shape[0] # batch size
    grid_shapes = [input_shape // {0:32,1:16,2:8}[l] for l in range(num_layers)] # 输出特征图的尺寸,相对于输入尺寸下采样32，16，8

    '''
    处理后的用于训练的数据
    y_true是个长度为3的列表，对应输出的3个尺度。其shape如下：
    (m,13,13,3,5+num_classes),(m,26,26,3,5+num_classes),(m,52,52,5 + num_classes)
    m : batch size
    13 , 26,52 : 网络层输出的feature map的大小， 也就是grid cell的大小
    3 : 每个 grid cell 预测3个anchor box
    5 : bbox相对于anchor box的偏移量及其置信度(x,y,w,h,confidence)，真实值confidenct = 1
    num_classes: bbox所属的类别
    '''
    y_true = [np.zeros((m,grid_shapes[1][0],grid_shapes[1][1],len(anchor_mask[l]),5 + num_classes),dtype=np.float32) for l in range(num_layers)]
    
    anchors = np.expand_dims(anchors,0) # 扩展第一个维度(1,9,2)
    anchors_maxes = anchors / 2. 
    anchors_mins = -anchors_maxes
    valid_mask = boxes_wh[...,0] > 0 # 判断是否有异常标注的boxes

    for b in range(m): # 每个batch size，也就是每张图片
        wh = boxes_wh[b,valid_mask[b]]
        if len(wh) == 0: continue

        wh = np.expand_dims(wh,-2) # 维度扩展(3,2) -> (3,1,2)
        box_maxes = wh / 2. 
        box_mins = -box_maxes

        # 当前的true box和哪个 anchor box的iou最大
        intersect_mins = np.maximum(box_mins,anchors_mins)
        intersect_maxes = np.minimum(box_maxes,anchors_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins,0)
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        box_area = wh[...,0] * wh[...,1]
        anchor_area = anchors[...,0] * anchors[...,1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou,axis=-1)
        # t,第几个true box; n, 和当前true box的iou最大的anchor box
        for t,n in enumerate(best_anchor): 
            for l in range(num_layers): # 第l个输出
                if n in anchor_mask[l] : # 不同的输出层有不同的anchor box，当前true box最匹配的anchor
                    '''
                    i,j: true box 映射到最后特征图上的某一个grid cell上，i为列，j为行
                    '''
                    i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype("int32")
                    j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype("int32")
                    # 每个grid cell有3个anchor box，k指和当前true box最匹配的索引[0,1,2]
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t,4].astype["int32"] # 类别
                    y_true[l][b,j,i,k,0:4] = true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4] = 1 # 置信度为1
                    y_true[l][b,j,i,k,5 + c] = 1

    return y_true

def get_random_data(annotation_line,input_shape,max_boxes=20):
    line = annotation_line.split()
    image = Image.open(line[0])
    iw,ih = image.size
    h,w = input_shape

    # true box
    box = np.array([np.array(list(map(int,box.split(",")))) for box in line[1:]])

    scale = min(np.float(w) / iw,np.float(h) / ih) # 等比缩放

    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw ) // 2
    dy = (h - nh) // 2
    image_data = 0

    image = image.resize((nw,nh),Image.BICUBIC)
    new_image = Image.new("RGB",(w,h),(128,128,128)) # 生成416 x 416的RGB图像
    new_image.paste(image,(dx,dy)) # 将resize后的图像放到416x416的图像中间
    image_data = np.array(new_image) / 255.0

    # correct boxes
    box_data = np.zeros(((max_boxes,5)))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes: box = box[:max_boxes] # 随机选择max_boxes box
        box[:,[0,2]] = box[:,[0,2]] * scale + dx # dx 为x轴的增量
        box[:,[1,3]] = box[:,[1,3]] * scale + dy # dy 为y轴的增量
        box_data[:len(box)] = box

    return image_data,box_data


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