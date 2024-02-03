from PIL import Image
import numpy as np
import rembg


def preprocess(image):
    with Image.open(image) as img:
        img = img.convert("L")
        img_array = np.array(img)
        output_image = rembg.remove(img_array, bgcolor=(255, 255, 255, 0))
        output_image = Image.fromarray(output_image)
        output_image = output_image.convert("RGB")
        output_image = output_image.resize((150, 150))

    return output_image
