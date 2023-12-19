from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2

def denoise_image(input_img_path, output_img_path):
    image_denoise_pipeline = pipeline(Tasks.image_denoising, 'damo/cv_nafnet_image-denoise_sidd')
    result = image_denoise_pipeline(input_img_path)[OutputKeys.OUTPUT_IMG]
    cv2.imwrite(output_img_path, result)

# img = 'noisy_image_gaussian_1.jpg'
# image_denoise_pipeline = pipeline(Tasks.image_denoising, 'damo/cv_nafnet_image-denoise_sidd')
# result = image_denoise_pipeline(img)[OutputKeys.OUTPUT_IMG]
# cv2.imwrite('result1.png', result)

if __name__ == "__main__":
    input_img_path = 'output_noise_image/noisy_image_gaussian_1.jpg'
    output_img_path = 'output_recovered_image/recover_image_gaussian_1.png'
    denoise_image(input_img_path, output_img_path)