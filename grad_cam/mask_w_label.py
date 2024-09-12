import os
from PIL import Image
import numpy as np


images_dir = './duke_images'
target_dir = './duke_images_mask_w_label'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
images = os.listdir(images_dir)
for image in images:
    current_image = os.path.join(images_dir, image)
    current_label = current_image.replace('duke_images','labels')
    current_mask = current_image.replace('duke_images','mask')
    # 以0-1的形式读取图像
    label =  Image.open(current_label)
    mask = Image.open(current_mask).convert('L')
    label_array = np.array(label) / 255
    mask_array = np.array(mask) / 255
    mask_array = mask_array * 0.5
    final_pesudo_label = (mask_array + label_array) * 50
    final_pesudo_label = Image.fromarray(final_pesudo_label.astype(np.uint8))
    final_pesudo_label.save(os.path.join(target_dir, image))


