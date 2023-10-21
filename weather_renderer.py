import os
import random

import skimage
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

def render_rain(img, theta):

    density = random.random()*0.5 # 密度
    intensity = random.random() # 强度

    # img: numpy ndarray format
    h, w = img.shape[: 2]
    img = np.power(img, 2)

    # parameter seed gen
    s = 1.01 + random.random() * 0.2
    m = density * (0.2 + random.random() * 0.05)  # mean of gaussian, controls density of rain
    v = intensity + random.random() * 0.3  # variance of gaussian,  controls intensity of rain streak
    length = random.randint(1, 40) + 20  # len of motion blur, control size of rain streak

    # Generate proper noise seed
    dense_chnl = np.zeros([h, w, 1])
    dense_chnl_noise = skimage.util.random_noise(dense_chnl, mode='gaussian', mean=m, var=v)
    dense_chnl_noise = cv2.resize(dense_chnl_noise, dsize=(0, 0), fx=s, fy=s)
    pos_h = random.randint(0, dense_chnl_noise.shape[0] - h)
    pos_w = random.randint(0, dense_chnl_noise.shape[1] - w)
    dense_chnl_noise = dense_chnl_noise[pos_h: pos_h + h, pos_w: pos_w + w]

    # form filter
    m = cv2.getRotationMatrix2D((length / 2, length / 2), theta - 45, 1)
    motion_blur_kernel = np.diag(np.ones(length))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (length, length))
    motion_blur_kernel = motion_blur_kernel / length
    dense_chnl_motion = cv2.filter2D(dense_chnl_noise, -1, motion_blur_kernel)
    dense_chnl_motion[dense_chnl_motion < 0] = 0
    dense_streak = (np.expand_dims(dense_chnl_motion, axis=2)).repeat(3, axis=2)

    # Render Rain streak
    tr = random.random() * 0.05 + 0.04 * length + 0.2
    img_rain = img + tr * dense_streak
    img_rain[img_rain > 1] = 1
    actual_streak = img_rain - img

    return img_rain, actual_streak

def add_haze(image, atmosphere, transmission_factor, depth_value):
    # 将输入图像转换为浮点数类型
    image = image.astype(np.float32) / 255.0

    # 估计透射率
    transmission = 1 - transmission_factor * depth_value

    # 限制透射率的范围在0到1之间
    transmission = np.clip(transmission, 0, 1)

    # 估计带有雾的图像
    hazy_image = np.empty_like(image)
    for channel in range(image.shape[2]):
        hazy_image[:, :, channel] = image[:, :, channel] * transmission + atmosphere[channel] * (1 - transmission)

    # 将像素值限制在0到1之间
    hazy_image = np.clip(hazy_image, 0, 1)

    # 将图像像素值转换回8位无符号整数类型（0-255）
    hazy_image = (hazy_image * 255).astype(np.uint8)

    return hazy_image

# 合成单张有雾图像
def synthesize_haze_image(input_image, output_path):
    image = cv2.imread(input_image)
    image_name = input_image.split("/")[-1]
    # 设置大气光、透射率系数和深度值
    atmos_num = random.uniform(0.3, 1)
    atmosphere = np.array([atmos_num, atmos_num, atmos_num])  # 大气光
    transmission_factor = random.uniform(0.5, 1.3)  # 透射率系数
    depth_value = 0.5  # 深度值
    hazy_image = add_haze(image, atmosphere, transmission_factor, depth_value)
    cv2.imwrite(output_path+image_name,hazy_image)
    return  hazy_image

# 将文件夹中所有图像合成为有雾图像
def synthesize_haze_images(imgs_dir,output_dir):
    # NOTE: 输入图像路径和输出图像路径后面要加"/"
    for im in tqdm(os.listdir(imgs_dir)):
        input_image = imgs_dir+im
        image = cv2.imread(input_image)
        image_name = input_image.split("/")[-1]
        # 设置大气光、透射率系数和深度值
        atmos_num = random.uniform(0.3, 1)
        atmosphere = np.array([atmos_num, atmos_num, atmos_num])  # 大气光
        transmission_factor = random.uniform(0.5, 1.3)  # 透射率系数
        depth_value = 0.5  # 深度值
        hazy_image = add_haze(image, atmosphere, transmission_factor, depth_value)
        cv2.imwrite(output_dir + image_name, hazy_image)

        # print(im)

def synthesize_rain_images(imgs_dir, output_dir):
    # 输入文件夹和输出文件夹后面要加 ‘/’
    for im in tqdm(os.listdir(imgs_dir)):
        image_name = im.split('/')[-1]
        img = Image.open(imgs_dir+im)
        img = np.array(img).astype(float) / 255
        rain_image,_ = render_rain(img, 180)
        # 将NumPy数组转换为图像对象
        img_rain = (rain_image * 255).astype(np.uint8)  # 将像素值转换为0-255范围内的整数
        image = Image.fromarray(img_rain)
        image.save(output_dir + image_name)
        # print(im)



if __name__ == '__main__':
    imgs_dir = r'G:\some_dataset\VOCdevkit\VOC2007\JPEGImages/'
    output_dir = r'G:\some_dataset\VOCdevkit\VOC2007\haze/'
    synthesize_haze_images(imgs_dir,output_dir=output_dir)
    # synthesize_rain_images(imgs_dir,output_dir=output_dir)