import cv2
import sys

#initialize
drawing = False
ix, iy = -1, -1
rect = (0, 0, 0, 0)

# 鼠标函数
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            temp_img = frame.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x-ix, y-iy)
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', frame)

# loading veido
cap = cv2.VideoCapture('test6.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# catch fisrt img of the vedio
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    sys.exit()

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #esc
        break
    elif k == 13: #Enter确认选择并退出
        break

#初始化KCF
tracker = cv2.TrackerKCF_create()
success = tracker.init(frame, rect)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 更新追踪器
    success, box = tracker.update(frame)
    if success:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

    cv2.imshow("Tracking", frame)

    # 退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
