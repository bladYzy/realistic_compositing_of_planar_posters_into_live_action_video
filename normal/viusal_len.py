import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image_path = 'test1/test1_frame.jpg'  # 修改为你的图片路径
img = mpimg.imread(image_path)

points = []


def onclick(event):
    if len(points) >= 2:
        return
    ix, iy = event.xdata, event.ydata
    print(f'Point selected: ({ix}, {iy})')
    points.append((ix, iy))

    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        print(f'The distance between the points is: {distance:.2f}')

fig, ax = plt.subplots()
ax.imshow(img)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
