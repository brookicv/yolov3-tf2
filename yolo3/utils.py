from functools import reduce
import numpy as np
import tensorflow as tf
from PIL import Image

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a,**kw)),funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")

def letterbox_image(image,size):
    """
    resize image with unchanged aspect ratio using padding
    """
    iw,ih = image.size
    w,h = size
    scale = min(w / iw,h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw,nh),Image.BICUBIC)
    new_image = Image.new("RGB",size,(128,128,128))
    new_image.paste(image,((w - nw) // 2,(h - nh) // 2))
    return new_image

def box_iou(b1,b2):
    
    b1 = tf.expand_dims(b1,-2)
    b1_xy = b1[...,:2]
    b1_wh = b1[...,2:4]
    b1_wh_half = b1_wh / 2.0
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2 = tf.exp(b2,0)
    b2_xy = b2[...,0:2]
    b2_wh = b2[...,2:4]
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