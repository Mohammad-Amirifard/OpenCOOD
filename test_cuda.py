import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Allow both GPU 0 and GPU 1

import torch

print("Number of visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
