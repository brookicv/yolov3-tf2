
import numpy as np
from PIL import Image
from PIL import ImageDraw

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
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5 + num_classes),dtype=np.float32) for l in range(num_layers)]

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
                    c = true_boxes[b,t,4].astype("int32") # 类别
                    y_true[l][b,j,i,k,0:4] = true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4] = 1 # 置信度为1
                    y_true[l][b,j,i,k,5 + c] = 1

                    print(b,j,i,k)

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

    ''' test codes 
    draw = ImageDraw.Draw(new_image)
    for b in box_data:
        draw.rectangle((b[0],b[1],b[2],b[3]),width=1,outline=(0,255,0))
    new_image.show()
    '''

    return image_data,box_data

def data_generator(annotation_lines,batch_size,input_shape,anchors,num_classes):
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0: 
                np.random.shuffle(annotation_lines) 
            
            image,box = get_random_data(annotation_lines[i],input_shape)
            image_data.append(image)
            box_data.append(box)

            i = (i + 1) % n # one epoch
        
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data,input_shape,anchors,num_classes)

        yield [image_data,*y_true],np.zeros(batch_size)

def data_generator_warpper(annotation_lines,batch_size,input_shape,anchors,num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: 
        return None
    return data_generator(annotation_lines,batch_size,input_shape,anchors,num_classes)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


if __name__ == '__main__':

    annotation_file = "head-dataset/VOCPartB/train.txt"
    anchors_file = "model_data/yolo_anchors.txt"
    classes_file = "model_data/classes.txt"

    with open(annotation_file) as f:
        lines = f.readlines()

    l = lines[0]

    image_data,box_data = get_random_data(l,(416,416))

    classes = get_classes(classes_file)
    anchors = get_anchors(anchors_file)

    box_data = np.expand_dims(box_data,0)
    y_true = preprocess_true_boxes(box_data,(416,416),anchors,len(classes))
    
    print(y_true[2][0,26,4,0])


