# attention_vgg16_cbam.py

import os
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dense,
    Multiply,
    Conv2D,
    Add,
    Activation,
    Reshape,
    Concatenate,
    Lambda,
    MaxPooling2D,
    Flatten,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the data directory and dataset path
data_dir = os.path.join(script_dir, "data")
dataset_path = os.path.join(data_dir, "imagenette2")

# Create the data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Dataset configuration
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
dataset_tar_path = os.path.join(data_dir, "imagenette2.tgz")

# Download and extract dataset if it doesn't exist
if not os.path.exists(dataset_path):
    print("Downloading and extracting the Imagenette dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_tar_path)
    with tarfile.open(dataset_tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    os.remove(dataset_tar_path)
    print("Download and extraction complete.")
else:
    print("Dataset already exists. Skipping download.")

# Define train and validation directories
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")

# Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

# Load VGG16 without the top layers (exclude default classification layers)
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


# Implement the CBAM module
def cbam_module(input_feature, reduction_ratio=16):
    """Convolutional Block Attention Module (CBAM)"""
    # Channel Attention Module
    channel = int(input_feature.shape[-1])
    shared_layer_one = Dense(
        channel // reduction_ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation("sigmoid")(channel_attention)
    channel_refined = Multiply()([input_feature, channel_attention])
    # Spatial Attention Module
    avg_pool_spatial = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(
        channel_refined
    )
    max_pool_spatial = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(
        channel_refined
    )
    spatial_attention = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = Conv2D(
        1,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(spatial_attention)
    spatial_refined = Multiply()([channel_refined, spatial_attention])
    return spatial_refined


# Function to freeze pretrained layers (VGG16 base layers)
def freeze_layers(model):
    for layer in model.layers:
        if layer.name.startswith("block"):
            # Freeze all layers that are part of VGG16
            layer.trainable = False
        else:
            # Leave CBAM and custom layers trainable
            layer.trainable = True


# Positions to insert CBAM
positions = {
    "input": "block1_conv1",
    "pool1": "block1_pool",
    "pool2": "block2_pool",
    "pool3": "block3_pool",
    "pool4": "block4_pool",
    "pool5": "block5_pool",
}


# Function to create model with CBAM inserted at a given position
def create_model_with_cbam(position):
    input_tensor = Input(shape=(224, 224, 3))
    x = input_tensor
    # Iterate over the layers in VGG16 (without top layers)
    for layer in vgg16.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        layer_config = layer.get_config()
        new_layer = layer.__class__.from_config(layer_config)
        # Call the new layer on the input to build it
        x = new_layer(x)
        # Set weights after the layer has been built
        if layer.get_weights():
            new_layer.set_weights(layer.get_weights())
        # Insert CBAM at the specified position
        if layer.name == position:
            x = cbam_module(x)
    # Add custom classification layers
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation="softmax")(x)  # Corrected line
    model = Model(inputs=input_tensor, outputs=x)
    return model


# Insert CBAM between input and conv1-1
def create_model_with_cbam_at_input():
    input_tensor = Input(shape=(224, 224, 3))
    x = cbam_module(input_tensor)
    # Rebuild the model from the first convolutional layer
    for layer in vgg16.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        layer_config = layer.get_config()
        new_layer = layer.__class__.from_config(layer_config)
        # Call the new layer on the input to build it
        x = new_layer(x)
        # Set weights after the layer has been built
        if layer.get_weights():
            new_layer.set_weights(layer.get_weights())
    # Add custom classification layers
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation="softmax")(x)  # Corrected line
    model = Model(inputs=input_tensor, outputs=x)
    return model


# Create models with CBAM at different positions
models_with_cbam = {}
models_with_cbam["input"] = create_model_with_cbam_at_input()

for key, position in positions.items():
    if key != "input":
        print(f"Creating model with CBAM at position: {key}")
        model = create_model_with_cbam(position)
        models_with_cbam[key] = model

# Freeze layers in each model
for key in models_with_cbam:
    freeze_layers(models_with_cbam[key])

# Training parameters
num_epochs = 5
train_steps = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Compile and train models
for key, model in models_with_cbam.items():
    print(f"\nTraining model with CBAM inserted at position: {key}\n")
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
    )
    # Save the model and history if needed
    model.save(f"model_cbam_{key}.h5")
    # You can also save the history object for later analysis

# Baseline model without CBAM
print("\nTraining baseline VGG16 model without CBAM\n")
vgg16_baseline = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# Add custom classification layers
x = vgg16_baseline.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(10, activation="softmax")(x)  # Corrected line
vgg16_baseline = Model(inputs=vgg16_baseline.input, outputs=x)

freeze_layers(vgg16_baseline)

vgg16_baseline.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history_baseline = vgg16_baseline.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)

# Compare results
print("\nEvaluating models...\n")
baseline_eval = vgg16_baseline.evaluate(validation_generator)
print(f"Baseline Validation Accuracy: {baseline_eval[1]}")

for key, model in models_with_cbam.items():
    eval_result = model.evaluate(validation_generator)
    print(f"Model with CBAM at {key} - Validation Accuracy: {eval_result[1]}")


# Feature Map Extraction and Visualization
def get_feature_maps(model, layer_name, input_image):
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    intermediate_output = intermediate_layer_model.predict(
        np.expand_dims(input_image, axis=0)
    )
    return intermediate_output


def plot_feature_maps(feature_maps, num_images=16):
    square = int(np.sqrt(num_images))
    fig, axes = plt.subplots(square, square, figsize=(12, 12))
    ix = 0
    for i in range(square):
        for j in range(square):
            if ix < feature_maps.shape[-1]:
                axes[i, j].imshow(feature_maps[0, :, :, ix], cmap="gray")
                axes[i, j].axis("off")
                ix += 1
            else:
                axes[i, j].axis("off")
    plt.show()


# Load a sample image from the validation set
sample_image, _ = validation_generator.next()
sample_image = sample_image[0]

# Visualize feature maps for each model
for key, model in models_with_cbam.items():
    # Find the name of the Multiply layer after CBAM for feature map extraction
    multiply_layers = [layer.name for layer in model.layers if "multiply" in layer.name]
    if multiply_layers:
        layer_name = multiply_layers[-1]
        feature_maps = get_feature_maps(model, layer_name, sample_image)
        print(f"\nFeature maps after CBAM at position {key}:\n")
        plot_feature_maps(feature_maps)
    else:
        print(f"No Multiply layer found in model with CBAM at position {key}")

# Feature maps for baseline model
# You can choose a layer to visualize, e.g., 'block5_conv3'
feature_maps_baseline = get_feature_maps(vgg16_baseline, "block5_conv3", sample_image)
print("\nFeature maps for baseline model:\n")
plot_feature_maps(feature_maps_baseline)
