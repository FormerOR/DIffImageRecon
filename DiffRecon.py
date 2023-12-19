import numpy as np
import os
import cv2
from ImageProcess import read_image, add_noise, save_image, show_image

def diffusion_model(noisy_image, num_iter, delta_t, kappa, option=1):
    # 如果图像是彩色的，对每个通道分别应用扩散模型
    if len(noisy_image.shape) > 2:
        diffused_image = list(cv2.split(noisy_image))
        for i in range(len(diffused_image)):
            diffused_image[i] = diffusion_model_single_channel(diffused_image[i], num_iter, delta_t, kappa, option)
        return cv2.merge(diffused_image)
    else:
        return diffusion_model_single_channel(noisy_image, num_iter, delta_t, kappa, option)

def diffusion_model_single_channel(noisy_image, num_iter, delta_t, kappa, option=1):
    diffused_image = noisy_image.copy()
    for i in range(num_iter):
        diffused_image = cv2.GaussianBlur(diffused_image, (3, 3), 0)
        gx, gy = np.gradient(diffused_image)
        norm_grad = np.sqrt(gx**2 + gy**2)
        diffused_image = diffused_image + delta_t * (cv2.Laplacian(diffused_image, cv2.CV_64F) - kappa * norm_grad)
    return diffused_image

def ReconImage():
    # TODO: 使用扩散模型恢复图像
    pass

if __name__ == "__main__":
    # 读取图像
    image_path = os.path.join("test_image", "image1.jpg")
    image = read_image(image_path)

    # 添加噪声
    noise_type = "gaussian"
    intensity = 1
    noisy_image = add_noise(image, noise_type, intensity)

    # 应用扩散模型
    num_iter = 5
    delta_t = 0.25
    kappa = 20
    recovered_image = diffusion_model(noisy_image, num_iter, delta_t, kappa)

    # 显示恢复的图像
    show_image(recovered_image)

    # 保存恢复的图像
    output_path = os.path.join("output_recovered_image", "recovered_image.jpg")
    save_image(recovered_image, output_path, noise_type, intensity)