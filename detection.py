import numpy as np

from tensorflow.keras.layers import Input
from PIL import Image

from yolo3.model import mobile_yolo_body
from yolo3.utils import letterbox_image
from yolo3.eval import  yolo_eval
from yolo3.dataset import get_anchors,get_classes

image_input = Input(shape=(416,416, 3))

model_body = mobile_yolo_body(image_input, 3, 1)
weights_path = "model/trained_weights_final.h5"
model_body.load_weights(weights_path)

image_path = "test.jpg"

image = Image.open(image_path) # channel order: R G B, OpenCV B G R

boxed_image = letterbox_image(image, tuple(reversed((416, 416)))) # resize to 416 x 416, letter mode.
image_data = np.array(boxed_image, dtype='float32')

# plt.imshow(boxed_image)
# plt.show()

img = image_data / 255. # 归一化处理 [0,1]
img = np.expand_dims(img, 0) # epxpand dims (416,416,3) -> (1,416,416,3)
img = img.astype("float32")

bboxes = model_body.predict(img)
# print(bboxes[0].shape)
# print(bboxes[1].shape)
# print(bboxes[2].shape)
# bboxes = model1.predict(tf.convert_to_tensor(img),steps=1,verbose=1)

anchors_file = "model_data/yolo_anchors.txt"
classes_file = "model_data/classes.txt"

anchors = get_anchors(anchors_file)

boxes,scores,classes = yolo_eval(bboxes,anchors,1,(1024,1023))
print(boxes)
