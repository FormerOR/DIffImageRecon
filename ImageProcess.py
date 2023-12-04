import cv2
import numpy as np
import os
# from ImageProcess import read_image, add_noise, save_image, show_image

def read_image(file_path):
    image = cv2.imread(file_path)
    return image

def add_noise(image, noise_type, intensity):
    noisy_image = np.copy(image)
    if noise_type == "gaussian":
        mean = 0
        std_dev = intensity
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
    elif noise_type == "salt_and_pepper":
        prob = intensity
        noise = np.random.choice([0, 255], size=image.shape[:2], p=[1 - prob, prob])
        noisy_image = cv2.add(image, noise[:, :, np.newaxis])
    return noisy_image

def show_image(image):
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
    cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)    # 窗口大小自适应图片大小
    # cv2.namedWindow('result', cv2.WINDOW_FREERATIO)   # 窗口大小自由调整
    # cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)   # 窗口大小保持图片大小
    # cv2.namedWindow('result', cv2.WINDOW_GUI_EXPANDED)    # 窗口大小自由调整，支持鼠标事件
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, file_path, noise_type, intensity):
    # Split the file_path into directory and file extension
    directory, file_extension = os.path.splitext(file_path)
    # Add noise_type and intensity to the file name
    new_file_path = f"{directory}_{noise_type}_{intensity}{file_extension}"
    cv2.imwrite(new_file_path, image)



if __name__ == "__main__":
    # 读取图像
    image_path = os.path.join("test_image", "image1.jpg")
    image = read_image(image_path)

    # 添加噪音
    noise_type = "gaussian"
    intensity = 1
    noisy_image = add_noise(image, noise_type, intensity)

    # 保存图像
    output_path = os.path.join("output_noise_image", "noisy_image.jpg")
    save_image(noisy_image, output_path, noise_type, intensity)

    # 展示图像
    show_image(noisy_image)