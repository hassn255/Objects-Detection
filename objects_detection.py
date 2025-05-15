import cv2
import os

class_file = "coco_class_labels.txt"
fp = open(class_file)
labels = fp.read().split('\n')

model_file = os.path.join("models","ssd_mobilenet_v2_coco_2018_03_29","frozen_inference_graph.pb")
config_file = os.path.join("models","ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

def detect_object(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim,dim), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 1

def display_text(im, text, x, y):
    textSize=cv2.getTextSize(text, FONTFACE,FONT_SCALE, THICKNESS)
    dim = textSize[0]
    base_line = textSize[1]
    cv2.rectangle(
        im,
        (x, y - dim[1] - base_line),
        (x + dim[0] , y + base_line),
        (0,0,0),
        cv2.FILLED
    )

    cv2.putText(
        im,
        text,
        (x, y-5),
        FONTFACE,
        FONT_SCALE,
        (0,255,255),
        THICKNESS,
        cv2.LINE_AA
    )

def display_objects(im, objects, scale_x , scale_y, threshold=0.1):
    for i in range(objects.shape[2]):
        classID = int(objects[0,0,i,1])
        score = float(objects[0,0,i,2])

        if score > threshold:
            x = int(objects[0,0,i,3] * 300 * scale_x  )
            y = int(objects[0,0,i,4] * 300 * scale_y )
            w = int((objects[0,0,i,5] * 300 * scale_x ) - x )
            h = int((objects[0,0,i,6] * 300 * scale_y ) - y )

            display_text(im , labels[classID] + " Confidence: %.2f"%score, x, y)
            cv2.rectangle(im, (x, y), (x + w , y + h), (255, 255, 255), 2, 1)

    return im

source = cv2.VideoCapture(0)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()

    if not has_frame:
        break

    frame = cv2.flip(frame,1)
    frame_resized = cv2.resize(frame, (300, 300))

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    scale_x = frame_width / 300
    scale_y = frame_height / 300

    objects = detect_object(net, frame_resized)
    img = display_objects(frame, objects,scale_x, scale_y, 0.5)
    cv2.imshow("camera preview", img)

source.release()
cv2.destroyAllWindows()