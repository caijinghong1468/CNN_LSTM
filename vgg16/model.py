import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

# 定義包含 VGG19 和 LSTM 的模型架構
class EmotionCNNLSTM(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNNLSTM, self).__init__()
        # 載入預訓練的 VGG19 模型作為卷積特徵提取器 (使用 ImageNet 上預訓練的權重)
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # 將 VGG19 的所有參數凍結 (不進行更新)，只使用其提取的特徵
        for param in vgg.parameters():
            param.requires_grad = False
        # 提取 VGG19 的卷積層部分 (features) 和平均池化層 (avgpool)
        self.cnn_features = vgg.features      # 卷積與池化層 (不包含最上層的全連接分類器)
        self.cnn_avgpool = vgg.avgpool        # 最後的 AdaptiveAvgPool2d 層，輸出固定大小的特徵圖

        # 定義兩層的 LSTM
        # input_size=512 對應每個時間步長的特徵向量維度 (VGG19 輸出通道數512)
        # hidden_size=512 表示每層 LSTM 隱藏狀態的維度
        # num_layers=2 則使用兩層堆疊的 LSTM 結構
        # batch_first=True 則輸入輸出張量的形狀為 (batch, seq_len, feature)
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)

        # 定義全連接層，將 LSTM 最終的隱藏狀態映射到情緒分類輸出 (7 類)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x 的輸入形狀為 (batch_size, 3, H, W)，圖像尺寸預期為 224x224x3
        # 通過 VGG19 的卷積網路提取特徵
        x = self.cnn_features(x)    # 經卷積與最大池化，輸出張量形狀: (batch_size, 512, H_feat, W_feat)
        x = self.cnn_avgpool(x)     # 自適應平均池化，將特徵圖大小統一為 (7, 7)
        # 現在 x 的形狀為 (batch_size, 512, 7, 7)

        # 將特徵圖展平成序列數據以輸入 LSTM
        # 先將空間維度展平: 把每張圖的 7x7 特徵圖攤平成 49 個位置的序列，每個位置有512維的特徵向量
        x = x.flatten(start_dim=2)           # 展平 H_feat 和 W_feat 維度，x 變為 (batch_size, 512, 49)
        x = x.permute(0, 2, 1)               # 調整維度順序為 (batch_size, 49, 512)，以符合 LSTM 輸入格式

        # 將展平後的特徵序列輸入 LSTM
        # lstm_out 輸出所有時間步的隱狀態序列，形狀為 (batch_size, seq_len, hidden_size)
        # (hn, cn) 分別是最終時間步每一層的隱狀態和記憶細胞狀態
        lstm_out, (hn, cn) = self.lstm(x)    # hn 形狀: (num_layers, batch_size, 512)

        # 取出最終時間步（序列最後一個位置）的輸出作為圖像的整體表示
        # 可以從 lstm_out 直接取，也可以使用 hn 提取最後一層隱狀態：
        final_feat = lstm_out[:, -1, :]      # 對應每個序列的最後一個時間步隱狀態，形狀 (batch_size, 512)

        # 經過全連接層將特徵轉換為每個情緒類別的分數
        out = self.fc(final_feat)            # 輸出形狀: (batch_size, num_classes=7)

        return out