import numpy as np
import os
from PIL import Image
import pandas as pd

# Create directory for images
if not os.path.exists("fake_images"):
    os.makedirs("fake_images")

# Number of fake images
num_images = 500

labels = []
img_paths = []

for i in range(num_images):
    # Create random noise image of shape 224x224 with 3 channels
    data = np.random.random((224, 224, 3)) * 255
    image = Image.fromarray(data.astype('uint8')).convert('RGB')
    
    # Save the image
    img_path = f"fake_images/img_{i}.jpg"
    image.save(img_path)
    img_paths.append(img_path)
    
    # Randomly assign a label (0: left, 1: right, 2: forward)
    label = np.random.randint(0, 3)
    labels.append(label)

# Create CSV for training (80%) and validation (20%)
train_size = int(0.8 * num_images)
val_size = num_images - train_size

train_data = {'img_path': img_paths[:train_size], 'label': labels[:train_size]}
val_data = {'img_path': img_paths[train_size:], 'label': labels[train_size:]}

train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)

train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("val_dataset.csv", index=False)
