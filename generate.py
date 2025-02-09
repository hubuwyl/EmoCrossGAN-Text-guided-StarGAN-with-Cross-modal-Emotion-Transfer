import os
import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
from model import Generator, Discriminator
from torchvision.utils import save_image
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_mapping = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def create_labels(batch_size, c_dim=5):
    """Generate target domain labels for debugging and testing."""
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(batch_size)*i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def denorm(x, is_uint=False):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    out.clamp_(0, 1)
    if is_uint:
        out *= 255
    return out
def load_image(img_path):
    """Load and preprocess an image."""
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法读取图像文件 {img_path}，请检查路径。")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    x_real = (image - 127.5) / 127.5
    x_real = torch.tensor(np.expand_dims(np.transpose(x_real, (2, 0, 1)), axis=0), dtype=torch.float32)
    x_real = x_real.to(device)
    return x_real

def main(config):
    g_conv_dim = 64
    c_dim = 7
    g_repeat_num = 6
    g_path = config.model

    # 检查模型文件是否存在
    if not os.path.exists(g_path):
        print(f"模型文件 {g_path} 不存在，请检查路径。")
        return

    g_model = Generator(g_conv_dim, c_dim, g_repeat_num)
    g_model.to(device)
    try:
        g_model.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))
        print(f"模型文件 {g_path} 加载成功")
    except Exception as e:
        print(f"加载模型时出现错误: {e}")
        return

    x_real = load_image(config.imgpath)
    if x_real is None:
        return

    with torch.no_grad():
        c_trg_list = create_labels(1, c_dim)
        if config.emotion is None:
            x_fake_list = []
            for n, c_trg in enumerate(c_trg_list):
                c_trg = c_trg.view(c_trg.size(0), c_trg.size(1), 1, 1)
                c = c_trg.repeat(1, 1, x_real.size(2), x_real.size(3))
                x = torch.cat([x_real, c], dim=1)
                x_fake = g_model(x_real, c_trg)
                x_fake_list.append(x_fake)
            x_concat = torch.cat(x_fake_list, dim=3)
            img_save = x_concat
            result_path = os.path.join('stargan/results/', 'output.jpg')
        elif config.emotion in emotion_mapping:
            emotion_index = emotion_mapping[config.emotion]
            c_trg = c_trg_list[emotion_index]
            c_trg = c_trg.view(c_trg.size(0), c_trg.size(1), 1, 1)
            c = c_trg.repeat(1, 1, x_real.size(2), x_real.size(3))
            x = torch.cat([x_real, c], dim=1)
            x_fake = g_model(x_real, c_trg)
            img_save = x_fake
            result_path = os.path.join('stargan/results/', f'output_{config.emotion}.jpg')
        else:
            print(f"不支持的表情: {config.emotion}")
            return

        # 检查保存目录是否存在，不存在则创建
        result_dir = 'stargan/results/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        try:
            save_image(denorm(img_save.data.cpu()), result_path, nrow=1, padding=0)
            print(f'Saved images into {result_path}...')
        except Exception as e:
            print(f"保存图像时出现错误: {e}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', type=str, default=None, help='Generating a photo with the relevant emoiton(angry,fear,disgust,surprise,sad,happy)')
    parser.add_argument('--model',type=str,default='stargan/models/180000-G.ckpt',help='load the model from this dir(.ckpt)')
    parser.add_argument('--imgpath',type=str,default='images/test_0164_aligned.jpg',help='path of imgs')
    config = parser.parse_args()
    print(config)
    main(config)


