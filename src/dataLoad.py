import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BASE_DIRECTORY = 'faces'

def process_data():
    image_data = []
    targets = []

    for i, folder in enumerate(os.listdir(BASE_DIRECTORY)):
        folder_path = os.path.join(BASE_DIRECTORY, folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            with Image.open(image_path) as img:
                img = img.resize((150, 150))
                img_array = np.array(img)
                image_data.append(img_array)
                targets.append(i)



    image_data = np.array(image_data)
    targets = np.array(targets)
    modified_data = image_data.reshape(165, 22500)
    train_data, test_data, train_label, test_label = [], [], [], []

    for person in range(15):
        person_indices = np.where(targets == person)[0]
        train_indices = person_indices[3:11]
        test_indices = person_indices[:3]
        train_data.extend(modified_data[train_indices])
        train_label.extend(targets[train_indices])
        test_data.extend(modified_data[test_indices])
        test_label.extend(targets[test_indices])

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    # print(train_data.shape)
    # for i in range(10):
    #     plt.imshow(train_data[i].reshape(120, 120))
    #     plt.title(train_label[i])
    #     plt.axis('off')
    #     plt.show()

    # for i in range(10):
    #     plt.imshow(test_data[i].reshape(120, 120))
    #     plt.title(test_label[i])
    #     plt.axis('off')
    #     plt.show()
    print()
    return train_data, train_label, test_data, test_label

process_data()