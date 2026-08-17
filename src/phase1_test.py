from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

results = model("test.jpg")

# Save annotated result
results[0].save(filename="output/phase1_result.jpg")

# Display result
results[0].show()