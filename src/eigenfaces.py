from dataLoad import process_data
import matplotlib.pyplot as plt
import numpy as np

train_data, train_label, test_data, test_label = process_data()
mean_face = np.mean(train_data, axis=0)
print(mean_face)
plt.imshow(mean_face.reshape(150, 150), cmap='gray')
plt.title("mean face")
plt.axis('off')
plt.show()