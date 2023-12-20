from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QSlider, QGridLayout, QHBoxLayout, QVBoxLayout, QFrame, QCheckBox, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from qt_material import apply_stylesheet

import os
import time
import threading
import ImageProcess as ip
import ModelAgent
import threading

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '基于扩散模型(Diffusion Models)的图像恢复'
        self.initUI()
        self.batch_size = 1
        self.diffusion_steps = 1000
        self.finished = 0
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.zoom = 2
        self.resize(1000*self.zoom,618*self.zoom)
        
        '''
        加载图片板块
        '''
        # 选择一张图片的按钮
        self.button = QPushButton('选择图片', self)
        self.button.resize(100, 38)
        self.button.clicked.connect(self.open_image)

        # 加载图片的label
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        '''
        添加噪音板块
        '''
        # 创建一个按钮用于添加噪音
        self.noise_button = QPushButton('添加噪音', self)
        self.noise_button.resize(100, 38)
        self.noise_button.clicked.connect(self.add_noise)

        # 创建一个label用于显示添加噪音后的图像
        self.noise_label = QLabel(self)
        self.noise_label.setAlignment(Qt.AlignCenter)

        # 在滑块上方创建一个文本提示调节噪音强度
        self.tip1 = QLabel(self)
        self.tip1.setText('调节噪音强度:')

        # 创建一个滑动条用于调整噪音强度
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged[int].connect(self.change_value)
        self.slider_value = QLabel(self)
        
        '''
        图像重建板块
        '''
        # 创建一个按钮用于图像重建
        self.reconstruction_button = QPushButton('图像重建', self)
        self.reconstruction_button.resize(100, 38)
        self.reconstruction_button.clicked.connect(self.reconstruction_thread)
        # self.reconstruction_button.clicked.connect(self.check_finished_thread)

        # 在滑块上方创建一个文本提示调节batch_size
        self.tip2 = QLabel(self)
        self.tip2.setText('设置batchsize(1-32):')

        # 创建一个滑动条用于调整batch_size的值
        self.batch_size_slider = QSlider(Qt.Horizontal, self)
        self.batch_size_slider.setRange(1, 32)
        self.batch_size_slider.setFocusPolicy(Qt.NoFocus)
        self.batch_size_slider.valueChanged[int].connect(self.change_batch_size)
        self.batch_size_value = QLabel(self)

        # 创建一个label用于显示重建后的图像
        self.reconstruction_label = QLabel(self)
        self.reconstruction_label.setAlignment(Qt.AlignCenter)

        # 在滑块上方创建一个文本提示调节diffusion_steps
        self.tip3 = QLabel(self)
        self.tip3.setText('设置diffusion_steps(50-5000):')

        # 再创建一个滑块用于调整diffuiosion_steps的值
        self.diffusion_steps_slider = QSlider(Qt.Horizontal, self)
        self.diffusion_steps_slider.setRange(50, 5000)
        self.diffusion_steps_slider.setFocusPolicy(Qt.NoFocus)
        self.diffusion_steps_slider.valueChanged[int].connect(self.change_diffusion_steps)
        self.diffusion_steps_value = QLabel(self)

        # 第二和第三板块都新增一个label用于显示加入噪音和重建后的图像相对于原图片的psnr和ssim
        self.psnr_label2 = QLabel(self)
        self.ssim_label2 = QLabel(self)
        self.psnr_label3 = QLabel(self)
        self.ssim_label3 = QLabel(self)



        '''
        布局添加
        '''
        # 创建一个布局并添加部件
        self.layout1 = QHBoxLayout()
        self.layout1_1 = QVBoxLayout()
        self.layout1_2 = QVBoxLayout()
        self.layout1_3 = QVBoxLayout()
        self.frame1 = QFrame()
        self.frame2 = QFrame()
        self.frame3 = QFrame()
        self.layout1.addWidget(self.frame1,1)
        self.layout1.addWidget(self.frame2,1)
        self.layout1.addWidget(self.frame3,1)

        self.frame1.setLayout(self.layout1_1)
        self.frame2.setLayout(self.layout1_2)
        self.frame3.setLayout(self.layout1_3)
        self.layout1_1.addWidget(self.button,1)
        self.layout1_1.addWidget(self.label,1)
        self.layout1_2.addWidget(self.noise_button,1)
        self.layout1_2.addWidget(self.tip1,1)
        self.layout1_2.addWidget(self.slider,1)
        self.layout1_2.addWidget(self.slider_value,1)
        self.layout1_2.addWidget(self.noise_label,10)
        self.layout1_3.addWidget(self.reconstruction_button,1)
        self.layout1_3.addWidget(self.tip2,1)
        self.layout1_3.addWidget(self.batch_size_slider,1)
        self.layout1_3.addWidget(self.batch_size_value,1)
        self.layout1_3.addWidget(self.tip3,1)
        self.layout1_3.addWidget(self.diffusion_steps_slider,1)
        self.layout1_3.addWidget(self.diffusion_steps_value,1)
        self.layout1_3.addWidget(self.reconstruction_label,12)

        # 将第二和第三板块新增的label添加到布局中
        self.layout1_2.addWidget(self.psnr_label2,1,Qt.AlignBottom | Qt.AlignCenter)
        self.layout1_2.addWidget(self.ssim_label2,1,Qt.AlignBottom | Qt.AlignCenter)
        self.layout1_3.addWidget(self.psnr_label3,1,Qt.AlignBottom | Qt.AlignCenter)
        self.layout1_3.addWidget(self.ssim_label3,1,Qt.AlignBottom | Qt.AlignCenter)
        
        # 创建三个标签用于标明阶段
        self.label1 = QLabel(self)
        self.label2 = QLabel(self)
        self.label3 = QLabel(self)
        self.label1.setText('STAGE 1: 加载图片')
        self.label2.setText('STAGE 2: 添加噪音')
        self.label3.setText('STAGE 3: 图像重建')

        # 设置三个标签的样式，字体放大加粗
        self.label1.setStyleSheet("font: 75 25pt \"黑体\";border: 2px solid white;")
        self.label2.setStyleSheet("font: 75 25pt \"黑体\";border: 2px solid white;")
        self.label3.setStyleSheet("font: 75 25pt \"黑体\";border: 2px solid white;")

        # 将三个标签添加到布局中，并且放置于布局底部并居中
        self.layout1_1.addWidget(self.label1,1,Qt.AlignBottom | Qt.AlignCenter)
        self.layout1_2.addWidget(self.label2,1,Qt.AlignBottom | Qt.AlignCenter)
        self.layout1_3.addWidget(self.label3,1,Qt.AlignBottom | Qt.AlignCenter)

        # 设置其他标签的文字大小
        self.slider_value.setStyleSheet("font: 75 10pt \"黑体\";")
        self.batch_size_value.setStyleSheet("font: 75 10pt \"黑体\";")
        self.diffusion_steps_value.setStyleSheet("font: 75 10pt \"黑体\";")
        self.tip1.setStyleSheet("font: 75 10pt \"黑体\";")
        self.tip2.setStyleSheet("font: 75 10pt \"黑体\";")
        self.tip3.setStyleSheet("font: 75 10pt \"黑体\";")

        # 设置按钮文字大小
        self.button.setStyleSheet("font: 75 13pt \"黑体\";")
        self.noise_button.setStyleSheet("font: 75 13pt \"黑体\";")
        self.reconstruction_button.setStyleSheet("font: 75 13pt \"黑体\";")

        # 为所有的frame添加边框
        self.frame1.setFrameShape(QFrame.StyledPanel)
        self.frame2.setFrameShape(QFrame.StyledPanel)
        self.frame3.setFrameShape(QFrame.StyledPanel)

        self.setLayout(self.layout1) 

    def open_image(self):
        image_path, _ = QFileDialog.getOpenFileName()
        image = ip.read_image(image_path)
        resized_image = ip.resize_image(image, 64, 64)
        self.resized_image_path = ip.save_cached_image(resized_image, "resized_image1.jpg")
        self.label.setPixmap(QPixmap(self.resized_image_path).scaled(64*self.zoom*4, 64*self.zoom*4))
        self.image_path = image_path  # Store the image path for noise addition
        print(image_path)

    def add_noise(self):
        image = ip.read_image(self.resized_image_path)
        noise_type = "gaussian"
        intensity = self.slider.value()
        noisy_image = ip.add_noise(image, noise_type, intensity)
        self.noise_path = ip.save_image(noisy_image, noise_type, intensity)
        self.abs_noise_path = os.path.abspath(self.noise_path)
        print(self.abs_noise_path)

        # 加载添加噪音后的图像
        self.noise_label.setPixmap(QPixmap(self.noise_path).scaled(64*self.zoom*4, 64*self.zoom*4))

        # 计算psnr和ssim
        psnr, ssim = ip.cal_psnr_ssim(ip.read_image(self.resized_image_path), noisy_image)
        self.psnr_label2.setText(f'峰值信噪比PSNR: {psnr}')
        self.ssim_label2.setText(f'结构相似性SSIM: {ssim}')    
        self.psnr_label2.setStyleSheet("font: 75 10pt \"黑体\";border: 2px solid white;")
        self.ssim_label2.setStyleSheet("font: 75 10pt \"黑体\";border: 2px solid white;")    

    def change_value(self, value):
        self.slider_value.setText(f'Noise level: {value}')

    def change_batch_size(self, value):
        self.batch_size_value.setText(f'Batch size: {value}')
        self.batch_size = value

    def change_diffusion_steps(self, value):
        self.diffusion_steps_value.setText(f'Diffusion steps: {value}')
        self.diffusion_steps = value

    def check_finished(self):
        while True:
            # 暂停1秒
            time.sleep(1)

            # 检查finished的值
            if self.finished == 1:
                # 创建一个消息框
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)

                # 设置消息框的标题和文本
                msg.setWindowTitle("提示")
                msg.setText("重建完成")

                # 显示消息框
                msg.exec_()

                # 重建完成后，退出循环
                break

    def reconstruction(self, batch_size=1):
        self.finished = ModelAgent.run_script(self.abs_noise_path, self.batch_size, self.diffusion_steps)
        output_path = f'E:/DIffImageRecon/DiffusionReconModel/out_image/final_image_diffusion_step_{self.diffusion_steps}.png'
        
        # 计算psnr和ssim
        psnr, ssim = ip.cal_psnr_ssim(ip.read_image(self.resized_image_path), ip.read_image(output_path))
        self.psnr_label3.setText(f'峰值信噪比PSNR: {psnr}')
        self.ssim_label3.setText(f'结构相似性SSIM: {ssim}') 
        self.psnr_label3.setStyleSheet("font: 75 10pt \"黑体\";border: 2px solid white;")
        self.ssim_label3.setStyleSheet("font: 75 10pt \"黑体\";border: 2px solid white;")     

        try:
            self.reconstruction_label.setPixmap(QPixmap(output_path).scaled(64*self.zoom*4, 64*self.zoom*4))
        except:
            print('load failed')
            self.reconstruction_label.setPixmap(QPixmap(self.image_path).scaled(64*self.zoom*4, 64*self.zoom*4))

    def reconstruction_thread(self):
        with self.lock:
            thread = threading.Thread(target=self.reconstruction)
            thread.start()

    def check_finished_thread(self):
        with self.lock2:
            thread = threading.Thread(target=self.check_finished)
            thread.start()
    
    
if __name__ == '__main__':
    app = QApplication([])
    # apply_stylesheet(app, theme='dark_teal.xml')
    # apply_stylesheet(app, theme='light_amber.xml')
    # apply_stylesheet(app, theme='dark_pink.xml',)
    # apply_stylesheet(app, theme='dark_purple.xml',)
    # apply_stylesheet(app, theme='dark_red.xml',)
    apply_stylesheet(app, theme='dark_cyan.xml',)
    ex = App()
    ex.show()
    app.exec_()
