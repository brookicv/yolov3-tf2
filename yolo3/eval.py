
import tensorflow as tf
from yolo3.model import  yolo_head

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
    return boxes

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
        _boxes,_box_scores = yolo_boxes_and_scores(yolo_outputs[l],anchors[anchor_mask[l]],num_classes,input_shape,image_shape)
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