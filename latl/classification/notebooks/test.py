from super_gradients.training import models

model = models.get("yolo_nano", pretrained_weights="coco")
model.export("yolo_x_nano.onnx")
