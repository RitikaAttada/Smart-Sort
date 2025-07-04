from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from tensorflow.keras.preprocessing import image

# Define augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

def augment_images(folder, num_augmented_images):
    save_dir = folder + "_augmented"
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        # Generate augmented images
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix="aug", save_format="jpg"):
            i += 1
            if i >= num_augmented_images:
                break

# Apply augmentation to "Wet" and "Other" # Generate 5 new images per existing image
augment_images("dataset/train/other", 2)
