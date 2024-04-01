import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 给定的三维向量
vector = [1.94520562e-05, 1.02084163e-03, 9.99997622e-01]

# 创建一个新的3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制向量
# 从原点到给定的点
ax.quiver(0, 0, 0, vector[0], vector[1], vector[2])

# 设置图形的坐标轴范围
ax.set_xlim([0, max(vector)])
ax.set_ylim([0, max(vector)])
ax.set_zlim([0, max(vector)])

# 添加标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
