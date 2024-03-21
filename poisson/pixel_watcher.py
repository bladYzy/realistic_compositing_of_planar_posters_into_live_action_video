import tkinter as tk
from PIL import Image, ImageTk

class ImageSelectApp:
    def __init__(self, master, image_path):
        self.master = master
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(master, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        # Save the starting point coordinates
        self.start_x = event.x
        self.start_y = event.y

        # Create a rectangle (at start, it is just a point)
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # Update the rectangle's coordinates to expand or move it
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        # Calculate the rectangle's width and height
        width = abs(event.x - self.start_x)
        height = abs(event.y - self.start_y)

        print(f"Selected rectangle dimensions: {width}x{height} pixels")

def main():
    root = tk.Tk()
    root.title("Image Selection App")
    app = ImageSelectApp(root, "source1.jpg")  # Replace "your_image_path_here.jpg" with your image file path
    root.mainloop()

if __name__ == "__main__":
    main()
