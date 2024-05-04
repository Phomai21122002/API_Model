import cv2
import numpy as np
from io import BytesIO

# Load YOLO
net = cv2.dnn.readNet("./models/yolov3.weights", "./models/yolov3.cfg")
classes = []
with open("./models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers, int):  
    unconnected_layers = [unconnected_layers]  

output_layers = [layer_names[i - 1] for i in unconnected_layers]

def DetectAnimal(image_data):
    imagesDetected = []
    if isinstance(image_data, str):  # If input is a file path
        img = cv2.imread(image_data)
    elif isinstance(image_data, BytesIO):  # If input is image data
        image_data.seek(0)  # Ensure we're at the start of the stream
        img_array = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        raise TypeError("Unsupported input type. Please provide a file path or image data.")

    if img is None:
        raise ValueError("Unable to read image. Please check the provided input.")

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cropped_image = img[max(0, y):min(y+h, height), max(0, x):min(x+w, width)]
            label = str(classes[class_ids[i]])
            if label in ['dog', 'cat']:
                imagesDetected.append(cropped_image)
    return imagesDetected