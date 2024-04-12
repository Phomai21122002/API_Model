import cv2
import numpy as np
import os


# Load YOLO
net = cv2.dnn.readNet("./models/yolov3.weights", "./models/yolov3.cfg")
classes = []
with open("./models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers, int):  # Kiểm tra nếu nó trả về một số nguyên
    unconnected_layers = [unconnected_layers]  # Chuyển đổi thành danh sách nếu là một số nguyên

output_layers = [layer_names[i - 1] for i in unconnected_layers]



def DetectAnimal(img):
    imagesDetected = []
    
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cropped_image = img[min(y, y+h):max(y, y+h), min(x, x+w):max(x, x+w)]
            label = str(classes[class_ids[i]])
            if(label == 'dog' or label =='cat'):               
                imagesDetected.append(cropped_image)
    return imagesDetected

       

