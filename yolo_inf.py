#Ultralytics file to understand how to use the YOLOv8 inference API

from ultralytics import YOLO

# Load a pretrained YOLOv8 model

model = YOLO('models/best.pt')  

results = model.predict('input_videos/08fd33_4.mp4', save=True)

print(results[0])
print("___________________________________________________")
for box in results[0].boxes:
    print(box)