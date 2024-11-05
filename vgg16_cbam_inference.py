# inference_image_saver.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import itertools


# Define the custom layers used in CBAM
class ChannelMean(Layer):
    def call(self, inputs):
        return K.mean(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class ChannelMax(Layer):
    def call(self, inputs):
        return K.max(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


# Dictionary of custom layers
custom_objects = {
    "ChannelMean": ChannelMean,
    "ChannelMax": ChannelMax,
}

# Define model positions
model_positions = ["input", "pool1", "pool2", "pool3", "pool4", "pool5", "baseline"]

# Directory where models are saved
models_dir = "models"

# Create directory to save inference results
results_dir = "inference_results"
os.makedirs(results_dir, exist_ok=True)

# Set up data generator
val_dir = os.path.join("data", "imagenette2", "val")
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

# Define the WordNet ID to class name mapping
wnid_to_class = {
    "n01440764": "tench",
    "n02102040": "English Springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# Retrieve class indices and create a mapping from index to WordNet ID
class_indices = validation_generator.class_indices
# Invert the dictionary to get a mapping from index to WordNet ID
idx_to_wnid = {v: k for k, v in class_indices.items()}

# Create a mapping from index to class name
idx_to_class = {
    idx: wnid_to_class.get(wnid, "Unknown") for idx, wnid in idx_to_wnid.items()
}

# Parameters for sampling
images_per_class = 5  # Number of images to sample per class


# Function to collect images per class
def collect_images_per_class(generator, images_per_class):
    """
    Collect a fixed number of images per class from the generator.

    Args:
        generator: Keras ImageDataGenerator iterator.
        images_per_class: Number of images to collect per class.

    Returns:
        A dictionary mapping class indices to lists of images and labels.
    """
    # Initialize dictionary to hold images per class
    images_dict = {i: {"images": [], "labels": []} for i in range(len(idx_to_class))}

    # Iterate over the generator
    for _ in range(len(generator)):
        batch_images, batch_labels = next(generator)
        for img, label in zip(batch_images, batch_labels):
            class_idx = np.argmax(label)
            if len(images_dict[class_idx]["images"]) < images_per_class:
                images_dict[class_idx]["images"].append(img)
                images_dict[class_idx]["labels"].append(label)

        # Check if we've collected enough images for all classes
        if all(
            len(info["images"]) >= images_per_class for info in images_dict.values()
        ):
            break

    # Flatten the dictionary into lists
    sampled_images = []
    sampled_labels = []
    for class_idx, info in images_dict.items():
        sampled_images.extend(info["images"])
        sampled_labels.extend(info["labels"])

    return np.array(sampled_images), np.array(sampled_labels)


# Collect sampled images and labels
sample_images, sample_labels = collect_images_per_class(
    validation_generator, images_per_class
)


# Function to save input images and predictions
def save_predictions(model, model_name, sample_images, sample_labels, idx_to_class):
    """
    Save images with predicted and true class annotations.

    Args:
        model: Trained Keras model.
        model_name: Name identifier for the model.
        sample_images: Numpy array of input images.
        sample_labels: Numpy array of true labels.
        idx_to_class: Dictionary mapping class indices to class names.
    """
    # Create directory for each model's results
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Perform predictions
    predictions = model.predict(sample_images)

    # Iterate over each image and save results
    for i, (img, prediction) in enumerate(zip(sample_images, predictions)):
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")

        # Get predicted class index and probability
        predicted_class_idx = np.argmax(prediction)
        predicted_prob = np.max(prediction)

        # Map index to class name
        predicted_class_name = idx_to_class.get(predicted_class_idx, "Unknown")

        # Get true class index and name
        true_class_idx = np.argmax(sample_labels[i])
        true_class_name = idx_to_class.get(true_class_idx, "Unknown")

        # Save prediction result in title
        pred_text = f"Pred: {predicted_class_name} ({predicted_prob*100:.2f}%)\nTrue: {true_class_name}"
        plt.title(pred_text, fontsize=8)

        # Save the image
        img_filename = (
            f"image_{i + 1}_pred_{predicted_class_name}_true_{true_class_name}.png"
        )
        img_path = os.path.join(model_dir, img_filename)
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()


# Load each model and save predictions
for position in model_positions:
    if position != "baseline":
        model_filename = f"model_cbam_{position}.h5"
    else:
        model_filename = "model_cbam_baseline.h5"

    model_path = os.path.join(models_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model = load_model(model_path, custom_objects=custom_objects)
        save_predictions(
            model=model,
            model_name=f"cbam_{position}",
            sample_images=sample_images,
            sample_labels=sample_labels,
            idx_to_class=idx_to_class,
        )
    else:
        print(f"Model file {model_path} not found. Skipping {position}.")
