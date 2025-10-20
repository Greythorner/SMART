#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, model_from_json

# ====================================================
# 1. Data Loading & Preprocessing
# ====================================================

# Load data (adjust the path to your file structure)
data = pd.read_excel('E:/SynologyDrive/Wuqiushuo/Morphology regeneration Data.xlsx', engine='openpyxl')

# Assume resistance (conditional) data and image paths exist
resistance_columns = [f'Resistance{i}' for i in range(1, 127)]
resistance_values = data[resistance_columns].values  # shape: (N, 126)

# Image preprocessing (no white noise)
def preprocess_image(image_path, size=(128, 128), color_mode='L'):
    """
    Load → resize → convert → normalize to [0,1].
    If loading fails, return a black image of target size.
    """
    try:
        image = Image.open(image_path)
        image = image.resize(size)
        image = image.convert(color_mode)
        image_array = np.array(image).astype(np.float32) / 255.0  # [0,1]
        if color_mode == 'L':
            image_array = np.expand_dims(image_array, axis=-1)
        return image_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros((size[0], size[1], 1), dtype=np.float32)

# Process images
image_paths = data['image_path'].values
image_batch = [preprocess_image(path) for path in image_paths]
real_images = np.array(image_batch)  # shape: (N, H, W, 1)

print(f"Real images range: min={real_images.min():.4f}, max={real_images.max():.4f}")

# Image & resistance dimensions
image_size = real_images.shape[1:]          # (H, W, C)
resistance_dim = resistance_values.shape[1] # 126

# ====================================================
# 2. Model Definitions
# ====================================================

from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, PReLU, Conv2DTranspose

def residual_block(x):
    """Residual block with a 1x1 projection to match channels."""
    in_x = x
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    in_x = Conv2D(64, (1, 1), padding='same')(in_x)
    x = Add()([in_x, x])
    return x

def build_generator(input_dim, condition_dim):
    noise_input = layers.Input(shape=(input_dim,))
    condition_input = layers.Input(shape=(condition_dim,))
    merged_input = layers.concatenate([noise_input, condition_input])

    x = layers.Dense(16 * 16 * 256, activation='relu')(merged_input)
    x = layers.Reshape((16, 16, 256))(x)

    for _ in range(16):
        x = residual_block(x)

    # Upsampling to 128x128
    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)  # 32x32
    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)  # 64x64
    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)  # 128x128

    output = layers.Conv2D(1, (3, 3), padding='same', activation='tanh')(x)
    return models.Model([noise_input, condition_input], output)

def build_discriminator(image_shape, condition_dim):
    image_input = layers.Input(shape=image_shape)
    condition_input = layers.Input(shape=(condition_dim,))

    # Map condition to spatial tensor and concat with image
    condition_layer = layers.Dense(image_shape[0] * image_shape[1], activation='relu')(condition_input)
    condition_layer = layers.Reshape((image_shape[0], image_shape[1], 1))(condition_layer)
    condition_layer = layers.Conv2D(image_shape[-1], (3, 3), padding='same')(condition_layer)

    x = layers.concatenate([image_input, condition_layer])

    for i in range(8):
        filters = 64 * min(2 ** (i // 2), 8)
        stride = 2 if i % 2 == 1 else 1
        x = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Re-inject condition after each downsampling (except the last)
        if i % 2 == 1 and i < 7:
            current_shape = x.shape[1:4]
            num_elements = int(current_shape[0] * current_shape[1] * current_shape[2])
            condition_layer_current = layers.Dense(num_elements, activation='relu')(condition_input)
            condition_layer_current = layers.Reshape(
                (int(current_shape[0]), int(current_shape[1]), int(current_shape[2]))
            )(condition_layer_current)
            x = layers.concatenate([x, condition_layer_current])

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model([image_input, condition_input], output)

# ====================================================
# 3. Compilation
# ====================================================

generator = build_generator(100, resistance_dim)  # noise dim = 100
discriminator = build_discriminator(image_size, resistance_dim)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.00005, 0.5), metrics=['accuracy'])

# Assemble GAN (freeze discriminator during generator updates)
discriminator.trainable = False
gan_input_noise = layers.Input(shape=(100,))
gan_input_resistance = layers.Input(shape=(resistance_dim,))
fake_image = generator([gan_input_noise, gan_input_resistance])
gan_output = discriminator([fake_image, gan_input_resistance])
gan = models.Model([gan_input_noise, gan_input_resistance], gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# ====================================================
# 4. Model Save/Load
# ====================================================

def load_model_from_files(model_structure_path, model_weights_path):
    """
    Load a Keras model from JSON (structure) and H5 (weights).
    """
    with open(model_structure_path, 'r') as json_file:
        model_structure = json_file.read()
    model = model_from_json(model_structure)
    model.load_weights(model_weights_path)
    return model

# Try loading a trained generator (if available)
generator_weights_path = "best_generator_model.h5"
try:
    generator = load_model(generator_weights_path)
    print("Generator model loaded successfully.")
except Exception:
    generator = build_generator(100, resistance_dim)
    try:
        generator.load_weights(generator_weights_path)
        print("Generator weights loaded successfully.")
    except Exception:
        print("No pre-trained generator found; using freshly initialized model.")

# ====================================================
# 5. Image Generation & Display
# ====================================================

def generate_images(generator_model, resistance_data, num_examples=5):
    """
    Generate images conditioned on resistance data.
    Returns generated images in [0,1] and indices of chosen conditions.
    """
    noise = np.random.normal(0, 1, (num_examples, 100))
    selected_idx = np.random.randint(0, resistance_data.shape[0], num_examples)
    selected_conditions = resistance_data[selected_idx]

    print(f"Noise sample (first vector): {noise[0]}")
    print(f"Condition sample (first row): {selected_conditions[0]}")

    gen_images = generator_model.predict([noise, selected_conditions], verbose=0)
    print(f"Generated images range before scaling: min={gen_images.min():.4f}, max={gen_images.max():.4f}")

    # Generator outputs tanh → map to [0,1]
    gen_images = (gen_images + 1.0) / 2.0
    print(f"Generated images range after scaling:  min={gen_images.min():.4f}, max={gen_images.max():.4f}")

    return gen_images, selected_idx

def display_comparison(real_images, gen_images, selected_idx, num_examples=5, save_folder="generated_images"):
    """
    Display real vs generated images; save generated images to disk.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"Real images range: min={real_images.min():.4f}, max={real_images.max():.4f}")

    plt.figure(figsize=(15, 6))
    for i in range(num_examples):
        # Real
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(real_images[selected_idx[i]].squeeze(), cmap='gray')
        plt.title("Real")
        plt.axis('off')

        # Generated
        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(gen_images[i].squeeze(), cmap='gray')
        plt.title("Generated")
        plt.axis('off')

        # Print and save generated image as PNG
        print(f"Generated image {i + 1} matrix:\n{gen_images[i].squeeze()}\n")
        gen_uint8 = (gen_images[i].squeeze() * 255).astype(np.uint8)
        Image.fromarray(gen_uint8).save(os.path.join(save_folder, f"generated_image_{i + 1}.png"))

    plt.tight_layout()
    plt.show()

# ====================================================
# 6. Simple Region-Based Evaluation
# ====================================================

def evaluate_images(real_images, gen_images, threshold=0.5):
    """
    Compare the number of connected white regions and total white area (binary).
    """
    num_samples = min(real_images.shape[0], gen_images.shape[0])
    num_diffs, area_diffs = [], []

    for i in range(num_samples):
        real_img = real_images[i].squeeze()
        gen_img  = gen_images[i].squeeze()

        real_binary = (real_img > threshold).astype(np.int32)
        gen_binary  = (gen_img  > threshold).astype(np.int32)

        real_label = label(real_binary, connectivity=2)
        gen_label  = label(gen_binary,  connectivity=2)

        real_num = int(real_label.max())
        gen_num  = int(gen_label.max())
        real_area = int(real_binary.sum())
        gen_area  = int(gen_binary.sum())

        num_diffs.append(abs(real_num - gen_num))
        area_diffs.append(abs(real_area - gen_area))

        if i < 3:
            print(f"Sample {i+1}: Real num={real_num}, Gen num={gen_num}, Real area={real_area}, Gen area={gen_area}")

    return float(np.mean(num_diffs)), float(np.mean(area_diffs))

# ====================================================
# 7. SSIM Evaluation (optional)
# ====================================================

def evaluate_images_ssim(real_images, gen_images, num_examples=5):
    """
    Compute SSIM over the first `num_examples` pairs.
    """
    ssim_scores = []
    for i in range(num_examples):
        real_img = real_images[i].squeeze()
        gen_img  = gen_images[i].squeeze()
        score = ssim(real_img, gen_img, data_range=max(1e-12, gen_img.max() - gen_img.min()))
        ssim_scores.append(score)
        print(f"Sample {i+1}: SSIM={score:.6f}")
    return float(np.mean(ssim_scores))

# ====================================================
# 8. Main
# ====================================================

if __name__ == "__main__":
    num_examples = 5

    # Use the original (non-noisy) condition for generation
    gen_images, selected_idx = generate_images(generator, resistance_values, num_examples=num_examples)

    # Compare real vs generated
    display_comparison(real_images, gen_images, selected_idx, num_examples=num_examples)

    # Region-based evaluation
    avg_num_diff, avg_area_diff = evaluate_images(real_images, gen_images, threshold=0.5)
    print(f"Average region-count difference: {avg_num_diff:.6f}")
    print(f"Average total-area difference:   {avg_area_diff:.6f}")

    # Optional SSIM
    avg_ssim = evaluate_images_ssim(real_images, gen_images, num_examples=num_examples)
    print(f"Average SSIM: {avg_ssim:.6f}")

