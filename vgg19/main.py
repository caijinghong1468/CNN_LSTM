import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader  # 導入資料加載器
from model import EmotionCNNLSTM               # 導入定義好的模型架構
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    # 檢查是否有可用的 GPU，若有則使用 GPU，否則使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置：", device)  # 輸出當前使用的裝置
    
    # 初始化模型並將其移動到計算裝置上
    model = EmotionCNNLSTM(num_classes=7).to(device)
    model = torch.nn.DataParallel(model)  # 啟用多 GPU
    model.to(device)
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()             # 交叉熵損失函數，用於多分類問題的標準損失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 優化器，學習率設定為 0.001

    # 設定訓練的輪數 (epochs)
    num_epochs = 150

    # 開始進行訓練迴圈
    for epoch in range(num_epochs):
        model.train()  # 將模型設定為訓練模式 (啟用 dropout/BN 等，在此模型中主要無影響，但遵循慣例)
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_loader:
            # 將批次資料移動到所選擇的裝置 (GPU 或 CPU)
            images = images.to(device)
            labels = labels.to(device)

            # 前向傳播：計算模型輸出
            outputs = model(images)
            # 計算損失值 (預測輸出與真實標籤之間的差異)
            loss = criterion(outputs, labels)

            # 清零先前梯度
            optimizer.zero_grad()
            # 反向傳播：計算梯度
            loss.backward()
            # 更新模型參數
            optimizer.step()

            # 累加損失值，之後取平均以監控訓練過程
            running_loss += loss.item()
        
        # 計算該 epoch 平均損失並輸出
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - 訓練損失: {avg_loss:.4f}")

    # 在訓練完成後，保存模型的參數至檔案，方便後續載入推理或測試
    model_path = "emotion_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型訓練完成，權重已保存至 {model_path}")



if __name__ == "__main__":
    main()


