
# Load the Pre-trained VGG16 Model
# import VGG16 from keras.applications and load the model with ImageNet weights, excluding the top (fully connected) layers.

from tensorflow.keras.applications import VGG16
from cv2 import imread
import os

# Load the VGG16 model, pre-trained weights, exclude top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Add Custom Layers
# Freeze the layers of the base model to prevent them from being updated during training, then add your custom layers on top for the classification task.
from tensorflow.keras import layers, models

# Freeze the base model
base_model.trainable = False

# Create a custom head for our network
model = models.Sequential([
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dense(1024, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(2, activation='softmax')  # Assuming two classes: Cat and Dog
])


# Compile and Train the Model
# Compile the model, specifying the loss function, optimizer, and metrics. Then, train the model with your dataset.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_data = {}
for root, dirs, files in os.walk('./train'):
    for file in files:

# function to apply train test split to all files in a directory, then relocate them appropriately
def split_and_relocate_files(directory, test_size=0.2):
    for root, folders, files in os.walk(directory):
        if files:
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(files, test_size=test_size)
            for file in train:
                src = os.path.join(root, file)
                dst = os.path.join(root, 'train', file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
            for file in test:
                src = os.path.join(root, file)
                dst = os.path.join(root, 'test', file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

# Assuming `train_images`, `train_labels` is your training dataset
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)


# Evaluate the Model
# After training, evaluate your model's performance on a separate test dataset to see how well it performs.
# Assuming `test_images`, `test_labels` is your testing dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


# Save and Load Your Model
# Finally, save your trained model for later use and load it whenever you need to classify new images.
model.save('cats_vs_dogs_model.h5')


# To load the model:
from tensorflow.keras.models import load_model

model = load_model('cats_vs_dogs_model.h5')

