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
    # 準備使用與訓練時相同的模型結構，並載入訓練後保存的權重
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = EmotionCNNLSTM(num_classes=7).to(device)
    model.load_state_dict(torch.load("./emotion_model.pth", map_location=device))
    model = torch.nn.DataParallel(model)  # 啟用多 GPU
    model.to(device)
    model.eval()  # 將模型設定為評估模式 (停用訓練中才有的隨機因素，如 dropout)

    # 在測試資料集上執行推理 (Inference)
    y_true = []  # 用於存放實際標籤
    y_pred = []  # 用於存放模型預測的標籤
    with torch.no_grad():  # 禁用梯度計算，加速推理
        for images, labels in test_loader:
            # 將資料移至適當裝置
            images = images.to(device)
            labels = labels.to(device)
            # 獲取模型的預測輸出
            outputs = model(images)
            # 取得預測的類別索引 (輸出最大值的索引)
            _, predicted = torch.max(outputs, dim=1)
            # 將結果移回 CPU 並轉為numpy，加入列表中
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 將收集到的預測和真實值轉為 numpy array (便於後續計算指標)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 計算準確率 (Accuracy)
    acc = accuracy_score(y_true, y_pred)
    # 計算精確率 (Precision)、召回率 (Recall)、F1-score
    # 使用 average='macro' 對多類別進行宏平均計算，統計整體表現
    precision = precision_score(y_true, y_pred, average='macro')
    recall    = recall_score(y_true, y_pred, average='macro')
    f1        = f1_score(y_true, y_pred, average='macro')

    # 輸出各項評估指標
    print(f"Accuracy (準確率): {acc:.4f}")
    print(f"Precision (精確率): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    # 定義類別名稱
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    # 繪製混淆矩陣圖表
    plt.figure(figsize=(8, 6))
    # 使用 Seaborn 繪製帶有色彩強度表示的熱力圖，顯示混淆矩陣
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')   # 標題：混淆矩陣
    plt.xlabel('Predicted Label')   # x軸標籤：模型預測的類別
    plt.ylabel('True Label')        # y軸標籤：真實類別
    plt.tight_layout()
    # **將圖表存成圖片**
    plt.savefig("confusion_matrix.png")  # 儲存為 PNG 圖檔
    plt.close()  # 關閉圖表，釋放記憶體


if __name__ == "__main__":
    main()