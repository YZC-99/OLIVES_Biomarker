from PIL import Image
import os

def convert_bmp_to_png(bmp_image_path, output_image_path):
    with Image.open(bmp_image_path) as img:
        img.save(output_image_path, 'PNG')

# Example usage:
bmp_image_path = '/home/gu721/yzc/OLIVES_Biomarker/grad_cam/sn3204_1.bmp'
output_image_path = '/home/gu721/yzc/OLIVES_Biomarker/grad_cam/sn3204_1.png'

convert_bmp_to_png(bmp_image_path, output_image_path)

print(f'Converted {bmp_image_path} to {output_image_path}')
