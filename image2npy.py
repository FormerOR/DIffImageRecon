import cv2
import numpy as np
import datetime
from ImageProcess import add_noise

def image2npy(image_path, npy_path):
   # 读取图片
    image = cv2.imread(image_path)
    # 转换为numpy数组
    image_array = np.array(image)
    img_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # 保存为npy文件
    np.save(npy_path, img_array_rgb)

def read_npy(npy_path):
    # 读取npy文件
    npy_image = np.load(npy_path)
    # 转换为图片
    image = cv2.cvtColor(npy_image, cv2.COLOR_RGB2BGR)
    # 展示图片
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def addnoise2npy(image_path, npy_path, noise_type, intensity):
    # 读取图片
    image = cv2.imread(image_path)
    # 转换为numpy数组
    image_array = np.array(image)
    img_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # 添加噪音
    noisy_image = add_noise(img_array_rgb, noise_type, intensity)
    # 保存为npy文件
    np.save(npy_path, noisy_image)

if __name__ == "__main__":
    # 设置时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # 设置图片路径
    image_path = './test_image/java.jpg'
    npy_path = f"./npy_image/npyImage_{timestamp}.npy"

    # # 转换为npy文件
    # image2npy(image_path, npy_path)
    # print("image转换为npy文件成功！")

    # 添加噪音并转换为npy文件
    noise_type = "gaussian"
    intensity = 1
    addnoise2npy(image_path, npy_path, noise_type, intensity)
    print("image添加噪音并转换为npy文件成功！")

    # 读取npy文件
    read_npy(npy_path)
    print("npy文件读取为image成功！")

