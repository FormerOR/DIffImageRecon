## 基于扩散模型（Diffusion Models）的图像恢复

![UI](UI.png)

### **内容和要求**

基于扩散模型构建一个图像恢复模块，在对原图加上噪声以后可以通过该模块恢复原始图像。开发对整个恢复过程进行演示的相关界面，并比较不同的系统参数对最终输出结果的影响。

### 设计思路

1. **环境准备**：
   * **确定Python的版本（推荐使用Python 3.6及以上）。**
   * **安装必要的库和框架，如NumPy用于数值计算，PIL或OpenCV用于图像处理，TensorFlow或PyTorch用于实现深度学习模型。**
2. **模块划分**：
   * **图像处理模块：负责读取、显示和保存图像，以及添加噪声等预处理操作。**
   * **扩散模型模块：实现基于扩散模型的图像恢复算法。**
   * **参数配置模块：管理用户输入的参数，控制扩散模型的行为。**
   * **用户界面模块：提供图形用户界面(GUI)，让用户可以交互式地使用系统。**
   * **结果评估模块：计算并展示图像恢复质量的指标，如PSNR和SSIM。**
3. **图像处理模块**：
   * **使用OpenCV或PIL读取用户上传的原始图像。**
   * **实现一个函数来在图像上添加不同类型和强度的噪声。**
   * **提供保存和展示图像的功能。**
4. **扩散模型模块**：
   * **选择合适的深度学习框架，如PyTorch或TensorFlow，来构建扩散模型。**
   * **根据扩散模型的理论，实现模型的前向传播和反向传播过程。**
   * **实现模型训练过程，包括数据加载、模型更新等。**
5. **参数配置模块**：
   * **设计参数输入界面，允许用户调整如迭代次数、步长等参数。**
   * **将用户输入的参数传递给扩散模型模块。**
6. **用户界面模块**：
   * **使用Tkinter或PyQt等库设计GUI。**
   * **实现文件上传按钮，让用户可以选择图像文件。**
   * **实现参数输入界面，让用户可以调整模型参数。**
   * **实现开始和停止按钮，控制图像恢复过程。**
   * **实现结果展示区域，动态展示图像从加噪到恢复的过程。**
7. **结果评估模块**：
   * **实现评估函数，计算恢复图像与原始图像之间的PSNR和SSIM等指标。**
   * **在GUI中展示评估结果，以供用户参考。**
8. **集成和测试**：
   * **将各个模块集成到一起，确保它们可以协同工作。**
   * **进行单元测试和集成测试，确保每个功能按预期工作。**
   * **进行用户测试，收集反馈并对系统进行迭代改进。**

### 开发顺序

#### **环境搭建**

* [X] **首先确保Python环境已经搭建好，并安装了OpenCV、PyTorch和PyQt等必要的库。**

#### **图像处理模块**

* [X] **用OpenCV开始基础的图像处理模块，包括读取、显示和保存图像的功能。**

  * [X] 读取图像 `read_image`
  * [X] 添加噪声 `add_noise`
  * [X] 保存图像 `save_image`
  * [X] 展示图像 `show_image`

#### **扩散模型原型**

* [X] **使用PyTorch构建扩散模型的初始版本。不需要GUI支持，可以专注于算法本身，确保它能够在简单的环境中运行和恢复图像。**

#### **参数配置和结果评估模块**

* [X] **在扩散模型能够基本工作之后，创建参数配置模块，允许调整模型参数。**
* [ ] **同时，开发结果评估模块，实现计算PSNR和SSIM等指标的功能，以便于监控模型性能。**

#### **用户界面原型**

* [X] **使用PyQt设计初步的用户界面原型。此时，你可以创建一个简单的窗口，包含必要的按钮和图像显示区域。**

#### **集成图像处理到用户界面**

* [X] **将图像处理模块集成到用户界面中，使得用户可以通过界面上传和显示图像。**

#### **模型和界面的集成**

* [X] **接下来，将扩散模型集成到用户界面中，允许用户通过界面对模型进行操作。**

#### **完善用户界面**

* [X] **根据扩散模型和图像处理模块的需求，完善用户界面，添加所有必要的控件和显示逻辑。**

#### **测试**

* [X] **对整个系统进行单元测试和集成测试，确保每个部分都能正常工作。**

> 需要重建图像的话，首先得下载 `imagenet64_uncond_100M_1500K.pt`这个训练好的模型放置在DiffusionReconModel文件夹内
