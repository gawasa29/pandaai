from ultralytics import YOLO

# Model Load
model = YOLO('./runs/detect/train/weights/last.pt')

# Model Use
model.predict('https://cdn.i-scmp.com/sites/default/files/d8/images/canvas/2022/01/26/795b5720-0a91-45cd-8f0c-ccfe494b1836_9d1e5b38.jpg', save=True, conf=0.2, iou=0.5)
