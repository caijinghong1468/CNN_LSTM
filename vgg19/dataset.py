# 導入必要的模組
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定義資料轉換 (Transforms)
# 訓練集使用數據增強與標準化：
# - 隨機水平翻轉圖像
# - 隨機旋轉圖像（±10度）
# - 將灰階影像轉為3通道 (因為預訓練的 VGG19 期望3通道輸入)
# - 調整圖像大小至 224x224 (符合 VGG19 輸入要求)
# - 轉換為 PyTorch 張量 (Tensor)
# - 正規化圖像 (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) 將像素值縮放到 [-1,1]
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # 以50%的機率隨機水平翻轉
    transforms.RandomRotation(degrees=10),    # 隨機旋轉 ±10 度內的角度
    transforms.Grayscale(num_output_channels=3),  # 將單通道灰階圖轉為3通道圖像
    transforms.Resize((224, 224)),            # 調整圖像大小至 224x224
    transforms.ToTensor(),                    # 轉換為張量並將像素值縮放到[0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化處理
])

# 測試集的轉換不需要隨機增強，只做必要的預處理和正規化：
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 將灰階轉為3通道
    transforms.Resize((224, 224)),                # 調整大小至 224x224
    transforms.ToTensor(),                        # 轉換為張量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化
])

# 使用 torchvision 提供的 FER2013 資料集類別讀取資料
# root: 資料存放的目錄，假設資料已下載並置於 ./data/fer2013
# split: 指定資料集劃分，"train" 為訓練集，"test" 為測試集
# transform: 指定上面定義的圖像轉換
train_dataset = datasets.FER2013(root='./data', split='train', transform=train_transform)
test_dataset  = datasets.FER2013(root='./data', split='test',  transform=test_transform)

# 建立資料加載器 DataLoader
# - batch_size: 每個批次處理的樣本數
# - shuffle: 訓練集隨機打亂順序，測試集不打亂
# - num_workers: 資料載入時使用的工作執行緒數量 (根據硬體情況調整，這裡設為2)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)

# 定義情緒類別名稱 (對應 FER2013 中的7種表情)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']