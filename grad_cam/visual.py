from models.resnet import  SupConResNet,LinearClassifier,LinearClassifier_MultiLabel
import torch
from pytorch_grad_cam import GradCAM,GradCAMPlusPlus
from PIL import Image
import cv2
import numpy as np
from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('svg')
import os
import copy


# 假设 grad_cam 是一个包含 GradCAM 结果的 numpy 数组，形状为 (H, W)
def grad_cam_to_mask(grad_cam, threshold=0.5):
    # 归一化 GradCAM 到 [0, 1]
    grad_cam = cv2.normalize(grad_cam, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 二值化处理
    _, binary_mask = cv2.threshold(grad_cam, threshold, 1, cv2.THRESH_BINARY)

    # 转换为 uint8 类型
    binary_mask = np.uint8(binary_mask * 255)

    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)

    # 应用闭运算（可以根据需要调整为其他形态学操作）
    # processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    processed_mask = binary_mask

    # 保留最大连通域，去除其他连通域，最后的掩码名字为processed_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
    max_label = 0
    max_size = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_size:
            max_label = i
            max_size = stats[i, cv2.CC_STAT_AREA]
    processed_mask = np.zeros_like(processed_mask)
    processed_mask[labels == max_label] = 255


    # # 将掩膜归一化到 [0, 1] 范围
    final_mask = processed_mask / 255.0
    # final_mask = processed_mask

    return final_mask

if __name__ == '__main__':

    model = 'resnet18'
    device = 'cuda'
    parallel=1
    ckpt_path="/home/gu721/yzc/OLIVES_Biomarker/Finetune_OCT_Clinical/unlock-layer4-fluid_srf/100_model.pth"
    # ckpt_path="/home/gu721/yzc/OLIVES_Biomarker/save/SupCon/Prime_TREX_DME_Fixed_models/patient_n_n_n_n_1_1_10_Prime_TREX_DME_Fixed_lr_resnet18_0.001_decay_0.0001_bsz_64_temp_0.07_trial_0__0/last.pth"
    model = SupConResNet(name=model)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        if parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.to(device)
        model.load_state_dict(state_dict)
    model.eval()
    targets = [ClassifierOutputTarget(0)]
    # cam = GradCAM(model=model, target_layers=model.encoder.layer4)
    cam = GradCAMPlusPlus(model=model, target_layers=model.encoder.layer4)

    grad2mask = True
    images_dir = './duke_images'
    target_dir = './duke_images_Finetune-mask-gradcam-overlabel'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    images = os.listdir(images_dir)
    for image in images:
        if image.endswith('.png'):
            current_image = os.path.join(images_dir, image)
            current_label = current_image.replace('duke_images','labels')
            label_array = np.array(Image.open(current_label)) / 255

            rgb_img = cv2.imread(current_image, 1)
            rgb_img = rgb_img.astype(np.float32) / 255

            img = read_image(os.path.join(images_dir, image))
            # input_tensor = (resize(img, (224, 224)) / 255).to(device)
            input_tensor = normalize(resize(img, (224, 224)) / 255, mean=[0.1904], std=[0.2088]).to(device)
            input_tensor = input_tensor.unsqueeze(0)
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets,aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
            if grad2mask:
                grayscale_cam = grad_cam_to_mask(grayscale_cam, threshold=0.38)
                # 保存为伪彩色图像
                # 红色是grayscale_cam, 绿色是label_array, 蓝色是grayscale_cam+label_array
                # 生成伪彩色图像
                final_pesudo_label = (grayscale_cam + label_array * 2)
                # 使用PIL 调色板保存伪彩色图像
                # 1是红色，2是绿色，3是蓝色，分别代表cam, label, cam+label
                final_pesudo_label = Image.fromarray(final_pesudo_label.astype(np.uint8))
                palette = [0, 0, 0,  # 黑色
                           255, 0, 0,  # 红色
                           0, 255, 0,  # 绿色
                           0, 0, 255]  # 蓝色
                # 扩展调色板至256色
                palette.extend([0, 0, 0] * (256 - len(palette) // 3))
                # 应用调色板
                final_pesudo_label.putpalette(palette)
                final_pesudo_label.save(os.path.join(target_dir, image))


                # plt.imshow(final_pesudo_label, cmap='gray')
                # plt.axis('off')
                # plt.savefig(os.path.join(target_dir, image), bbox_inches='tight', pad_inches=0)
            else:
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                plt.imshow(visualization)
                plt.axis('off')
                plt.savefig(os.path.join(target_dir, image), bbox_inches='tight', pad_inches=0)
