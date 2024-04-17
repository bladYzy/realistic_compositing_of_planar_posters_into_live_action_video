
v = [-0.78296846, -0.02731037,  0.6174835 ];

figure;

quiver3(0, 0, 0, v(1), v(2), v(3));

xlim([-10, 10]);
ylim([-10, 10]);
zlim([-10, 10]);

grid on;

title('三维向量可视化');
xlabel('X');
ylabel('Y');
zlabel('Z');
