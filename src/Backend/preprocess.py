import cv2
from PIL import Image
import numpy as np
import rembg


def detect_face(image_array):
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Get the coordinates of the first detected face
        x, y, w, h = faces[0]

        # Crop the face from the original image
        face = image_array[y:y+h, x:x+w]

        return face
    else:
        return None



def preprocess(image):
    with Image.open(image) as img:
        img = img.convert("L")
        img_array = np.array(img)
        output_image = rembg.remove(img_array, bgcolor=(255, 255, 255, 0))
        output_image = Image.fromarray(output_image)
        output_image = output_image.convert("RGB")
        output_image = output_image.resize((150, 150))

    return output_image
