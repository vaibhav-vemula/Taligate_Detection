import cv2
import numpy as np
import pandas as pd
from datetime import datetime


def personDetector(image):

    global lx,ly,bordercolor
    
    Width = image.shape[1]
    Height = image.shape[0]

        
    net1.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))
        
    person_layer_names = net1.getLayerNames()
    person_output_layers = [person_layer_names[i[0] - 1] for i in net1.getUnconnectedOutLayers()]
    person_outs = net1.forward(person_output_layers)
        
    person_class_ids = []
    person_confidences = []
    person_boxes = []

    for operson in person_outs:
        for detection in operson:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                person_class_ids.append(class_id)
                person_confidences.append(float(confidence))
                person_boxes.append([x, y, w, h])

    pindex = cv2.dnn.NMSBoxes(person_boxes, person_confidences, 0.5, 0.4)

    for i in pindex:
        i = i[0]
        box = person_boxes[i]
        lx=round(box[0]+box[2]/2)
        ly=round(box[1]+box[3])-10
        if person_class_ids[i]==0:
            label = str(coco_classes[person_class_ids[i]]) 
            cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), bordercolor, 2)
            cv2.putText(image, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, bordercolor, 2)
            cv2.circle(image, (lx,ly), radius=5, color=(255, 0, 0), thickness=-1)


def doorDetector(image):
    
    global xxx,yyy,www,hhh
    
    Width = image.shape[1]
    Height = image.shape[0]
    
    net2.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

    layer_names = net2.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net2.getUnconnectedOutLayers()]
    outs = net2.forward(output_layers)
    colors = np.random.uniform(0, 255, size=(len(door_class), 3))


    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            xxx, yyy, www, hhh = boxes[i]
            x, y, w, h = boxes[i]
            label = str(door_class[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 3, color, 2)
    


cap = cv2.VideoCapture('tailgate.mp4')
df = pd.read_csv('access_info.csv')
now = datetime.now()

coco_classes = None
with open('labels.txt', 'r') as f:
    coco_classes = [line.strip() for line in f.readlines()]

door_class = ['door']

net1 = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
net2 = cv2.dnn.readNet("yolov3_door.weights", "yolov3_door.cfg")

fps = 1
xxx,yyy,www,hhh = 0,0,0,0

lx=0
ly=0

cou=-1
bordercolor = (0,255,0)

id = int(input("ENTER EMPLOYEE ID  "))
print(' ')
emp_id = df[df["Employee ID"] == id]
inde = emp_id.index[0]

try:

    if(emp_id["Authorisation Result"].item() == 1):
        df.loc[inde,"Time of Swipe"] = now.strftime("%d/%m/%Y %H:%M:%S")

        while True:

            cap.set(cv2.CAP_PROP_POS_FRAMES, fps)
            _, image =cap.read()

            if fps < 30:
                doorDetector(image)
                

            personDetector(image)
            
            cv2.line(image, (xxx,yyy+hhh),(xxx+www+20,yyy+(hhh-40)), bordercolor, thickness=3)
            cv2.line(image, (xxx,yyy+hhh),(xxx,yyy+round(hhh/2)), bordercolor, thickness=3)
            cv2.line(image, (xxx+www+20,yyy+(hhh-40)),(xxx+www+10,yyy+round(hhh/2)), bordercolor, thickness=3)


            t1 = (lx - xxx)*((yyy+(hhh-40)) - (yyy+hhh))
            t2 = (ly - (yyy+hhh))*((xxx+www+20) - xxx)
            d = t1 - t2

            if d>0:
                cou+=1

            if cou >= 2:
                bordercolor = (0,0,255)
                df.loc[inde,"Tailgated"] = "YES"
                
                print("ALERT!! TAILGATING DETECTED")
                
            
            fps = fps + 5
            cv2.imshow("Result",image)
            key = cv2.waitKey(1)
            if key == 27:
                break


    else:
        print("INVALID ID\n")


except AttributeError:
    print("\nEND of VIDEO")
    

except Exception as e:
    print(e)


df.to_csv('access_info.csv',index=False)
cap.release()
cv2.destroyAllWindows()
