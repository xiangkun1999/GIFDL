from genericpath import exists
from importlib.resources import path
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import random
import numpy as np
import argparse
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guidance', dest='guidance', type=float, default=7.5)
    parser.add_argument(
        '--dir', dest='dir', type=str, default='/public/wangxiangkun/dataset_sd_in')
    args = parser.parse_args()
    return args


def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





args = parse_args()
if not os.path.exists(args.dir):
    os.mkdir(args.dir)


device = "cuda"


# 指定本地模型路径
local_model_path = "/public/wangxiangkun/liuruiheng/gen_sd_10000/stable-diffusion-v1-4"

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(local_model_path, revision="fp16", torch_dtype=torch.float16, use_auth_token='')


pipe = pipe.to(device)


# 读取
a=np.load('a.npy')
promptlist=a.tolist()

for i in range(10000):
    prompt = promptlist[i%1000]
    with autocast("cuda"):
        print(i)
        set_seed(i//1000+2024)
        # image = pipe(prompt, guidance_scale=args.guidance)["sample"][0]
        image = pipe(prompt, guidance_scale=args.guidance).images[0]
        image.save(args.dir+'/'+str(i+1)+".pgm")
