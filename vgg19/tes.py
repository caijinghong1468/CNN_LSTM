import torch

model_path = "./vgg19/emotion_model.pth"
state_dict = torch.load(model_path, map_location="cpu",weights_only=False)  # 先載入到 CPU，避免 CUDA 問題

print(type(state_dict))  # 檢查類型