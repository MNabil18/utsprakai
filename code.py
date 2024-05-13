import zipfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ekstraksi file zip
def extract_zip(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Path ke file zip dataset
zip_file_path = "hangul_dataset.zip"

# Path untuk mengekstrak dataset
extract_to = "dataset_folder"

# Ekstraksi zip
extract_zip(zip_file_path, extract_to)

# Load dataset
images = []  # List untuk menyimpan gambar
labels = []  # List untuk menyimpan label

# Loop melalui direktori yang berisi dataset yang telah diekstrak
for root, dirs, files in os.walk(extract_to):
    for file in files:
        if file.endswith(".png"):  # Ubah sesuai ekstensi gambar yang Anda miliki
            image_path = os.path.join(root, file)
            label = os.path.basename(root)  # Menggunakan nama direktori sebagai label
            labels.append(label)
            img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))  # Menggunakan tensorflow.keras
            img_array = tf.keras.preprocessing.image.img_to_array(img)  # Menggunakan tensorflow.keras
            images.append(img_array)

# Konversi menjadi array numpy
images = np.array(images)
labels = np.array(labels)

# Split dataset menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocessing data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(labels)), activation='softmax')  # Number of unique labels
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             fill_mode='nearest')

# Train the model
datagen.fit(X_train)
history = model.fit(datagen.flow(X_train, y_train_encoded, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=10, validation_data=(X_test, y_test_encoded))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)

print('Test accuracy:', test_acc)
