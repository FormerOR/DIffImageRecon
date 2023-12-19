import subprocess

def run_script(image_path, batch_size, diffusion_steps):
    # 用于在conda环境中执行指令的脚本
    script = f'activate base && cd DIffusionReconModel && python generate.py --model_path imagenet64_uncond_100M_1500K.pt --input_image {image_path} --batch_size {batch_size} --diffusion_steps {diffusion_steps}'

    # 使用subprocess执行脚本
    process = subprocess.Popen(script, shell=True)
    process.wait()
    return 1


if __name__ == '__main__':
    # 使用函数
    run_script('noise_java.jpg', 1)