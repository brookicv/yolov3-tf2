# yolov3-tf2

基于TensorFlow2实现的yolov3 ,包括训练部分，具体来说有以下功能：

- 不同的基础网络，包括`DarkNet-53`,`tiny-yolo`,`MobileNet`
- VOC格式转换为yolo格式
- GT-boxes的转换
- yolo-loss的实现
- 使用自定义的Lambda层实现yolo-head，便于部署
- 将训练后的模型转换为TF-Lite格式
