from PIL import Image
import os
import numpy as np
# Change this to your image filename
image_path = "019_CaseInformation_PlanandAnatomy.jpg"   # ←←← Change this

img = Image.open(image_path)
width, height = img.size
print(f"Image size: {width}x{height} pixels\n")
img = np.array(img)
# Upscale 4x (mirrors imresize(imag, 4))
h, w = img.shape[:2]
img = np.array(
    Image.fromarray(img).resize((w * 4, h * 4), Image.LANCZOS)
)

print("Click on the four corners of the box (Top-left, Top-right, Bottom-right, Bottom-left)")
print("Press Enter after each click.\n")

# This will open the image so you can click
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(img)

coords = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        coords.append((x, y))
        print(f"Point {len(coords)}: ({x}, {y})")
        if len(coords) == 4:
            plt.close()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Click the 4 corners of the box")
plt.show()

print("\nFinal corners:")
print("Top-left:", coords[0])
print("Top-right:", coords[1])
print("Bottom-right:", coords[2])
print("Bottom-left:", coords[3])