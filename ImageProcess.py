import cv2
import numpy as np
import os
import datetime
import cv2
from PyQt5.QtGui import QPixmap
# from ImageProcess import read_image, add_noise, save_image, show_image

def read_image(file_path):
    image = cv2.imread(file_path)
    return image

def add_noise(image, noise_type, intensity):
    noisy_image = np.copy(image)
    if noise_type == "gaussian":
        mean = 0
        std_dev = intensity/50
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
    elif noise_type == "salt_and_pepper":
        prob = intensity
        noise = np.random.choice([0, 255], size=image.shape[:2], p=[1 - prob, prob])
        noisy_image = cv2.add(image, noise[:, :, np.newaxis])
    return noisy_image

def show_image(image):
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
    # cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)    # 窗口大小自适应图片大小
    # cv2.namedWindow('result', cv2.WINDOW_FREERATIO)   # 窗口大小自由调整
    cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)   # 窗口大小保持图片大小
    # cv2.namedWindow('result', cv2.WINDOW_GUI_EXPANDED)    # 窗口大小自由调整，支持鼠标事件
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, file_path, noise_type, intensity):
    # Split the file_path into directory and file extension
    directory, file_extension = os.path.splitext(file_path)
    # Add noise_type and intensity to the file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_file_path = f"{directory}_{noise_type}_{intensity}_{timestamp}{file_extension}"
    cv2.imwrite(new_file_path, image)

def save_image(image, noise_type, intensity):
    # Create the output directory if it doesn't exist
    output_directory = "NoisyImage"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate the file name using noise_type, intensity, and system time
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{noise_type}_{intensity}_{timestamp}.jpg"
    file_path = os.path.join(output_directory, file_name)

    # Save the image
    cv2.imwrite(file_path, image)
    return file_path

    
def resize_image(image, width, height):
        resized_image = cv2.resize(image, (width, height))
        return resized_image

def save_cached_image(image, file_name):
    # Create the cacheIMG directory if it doesn't exist
    cache_directory = "cacheIMG"
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    # Generate the file path using the cache_directory and file_name
    file_path = os.path.join(cache_directory, file_name)

    # Save the image
    cv2.imwrite(file_path, image)
    return file_path



if __name__ == "__main__":
    # 读取图像
    image_path = os.path.join("test_image", "java.png")
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