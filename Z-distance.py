from ultralytics import YOLO
import cv2

H_mm = 300
f_mm = 1000

#loading YOLO model
model = YOLO("best.pt")

#reading the image
image = cv2.imread("image.jpg")

#object detection
results = model(image)

#storing depths
depth_list = []

#looping
for r in results:

    #bounding boxes
    boxes = r.boxes.xyxy

    for box in boxes:

        #bounding box coordinates
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        #pixel height of cone
        pixel_height = y2 - y1

        #pixel to metric conversion using formula
        depth_mm = (H_mm * f_mm) / pixel_height

        #convertion of mm to meters
        depth_m = depth_mm / 1000

        #save depth
        depth_list.append(depth_m)

        #draw bounding box
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

        #write distance on image
        text = "Dist: " + str(round(depth_m,2)) + "m"
        cv2.putText(image,text,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,255,0),2)

#print detected cones and their depths
print("Detected cones and their depths (in meters):")

for d in depth_list:
    print(d)

#save output image
cv2.imwrite("output.jpg",image)