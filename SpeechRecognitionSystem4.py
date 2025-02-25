# -coding: utf-8
# -Author: fengpengpei
# -Email: fpf0103@163.com
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 设置音频后端
# try:
#     torchaudio.set_audio_backend("soundfile")  # 或者 "sox"
# except RuntimeError as e:
#     print(f"无法设置音频后端: {e}")

# 显卡以及CUDA是否可用
if torch.cuda.is_available():
    print("GPU version installed.")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CPU version installed.")
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义情感标签映射
emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "ps": 6  # 惊喜
}

# 定义 MFCC 提取函数
def extract_mfcc(waveform, sample_rate=22050):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=20,  # MFCC 特征数量
        melkwargs={
            "n_fft": 512,  # FFT 点数
            "n_mels": 64,  # 梅尔滤波器组数量
            "hop_length": 256,  # 帧移
        }
    )
    mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)  # (time, n_mfcc)
    return mfcc.numpy()

# 定义重采样函数
def resample_if_necessary(waveform, target_sample_rate):
    if waveform.shape[1] != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=waveform.shape[1], new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform

# 定义混合声道函数
def mix_down_if_necessary(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

# 预处理并保存数据
def preprocess_and_save(data_dir, save_dir, sample_rate=22050):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_paths = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(data_dir, file_name)
            emotion = file_name.split("_")[-1].split(".")[0]
            if emotion in emotion_map:
                file_paths.append(file_path)
                labels.append(emotion_map[emotion])

    for i, file_path in enumerate(file_paths):
        # 加载音频文件
        waveform, _ = torchaudio.load(file_path)

        # 重采样
        waveform = resample_if_necessary(waveform, sample_rate)

        # 混合声道
        waveform = mix_down_if_necessary(waveform)

        # 提取 MFCC 特征
        mfcc = extract_mfcc(waveform, sample_rate)

        # 保存 MFCC 特征和标签
        save_path = os.path.join(save_dir, f"data_{i}.npy")
        np.save(save_path, {"mfcc": mfcc, "label": labels[i]})

        print(f"Processed {i + 1}/{len(file_paths)}: {file_path} -> {save_path}")

    # 保存文件路径和标签的映射
    np.save(os.path.join(save_dir, "file_paths.npy"), file_paths)
    np.save(os.path.join(save_dir, "labels.npy"), labels)

# 自定义数据集类（加载预处理后的数据）
class PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths = np.load(os.path.join(data_dir, "file_paths.npy"), allow_pickle=True)
        self.labels = np.load(os.path.join(data_dir, "labels.npy"), allow_pickle=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, f"data_{idx}.npy")
        data = np.load(data_path, allow_pickle=True).item()
        mfcc = data["mfcc"]
        label = data["label"]
        return mfcc, label

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 25, 128)  # 假设输入长度为 100
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度，形状为 [batch_size, 1, sequence_length]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out

# 定义 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_gru, _ = self.gru(x)
        out = self.fc(h_gru[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transformer 需要 (seq_len, batch, features)
        x = self.transformer_encoder(x)
        out = self.fc(x[-1, :, :])  # 取最后一个时间步的输出
        return out

# 训练模型
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 将模型移动到 GPU
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.float().to(device)  # 将数据移动到 GPU
            labels = labels.long().to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Accuracy: {100 * correct / total}%")

    return model

# GUI 界面
class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("语音情感识别系统")
        self.root.geometry("800x800")

        # 模型参数
        self.input_size = 20  # MFCC 特征维度
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = 7  # TESS 数据集有 7 种情感
        self.num_heads = 4  # Transformer 的注意力头数
        self.data_dir = "dataverseTESS"  # TESS 数据集路径
        self.preprocessed_dir = "preprocessed_data"  # 预处理数据保存目录

        # 创建 GUI 组件
        self.create_widgets()

    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="语音情感识别系统", font=("Arial", 20))
        title_label.pack(pady=10)

        # 模型选择（训练）
        self.train_model_var = tk.StringVar(value="LSTM")
        train_model_label = tk.Label(self.root, text="选择训练模型:", font=("Arial", 14))
        train_model_label.pack(pady=5)
        # train_model_menu = ttk.Combobox(self.root, textvariable=self.train_model_var, values=["LSTM", "CNN", "GRU", "Transformer"])
        train_model_menu = ttk.Combobox(self.root, textvariable=self.train_model_var,
                                        values=["LSTM", "GRU", "Transformer"])
        train_model_menu.pack(pady=5)

        # 预处理按钮
        preprocess_button = tk.Button(self.root, text="预处理数据", command=self.preprocess_data)
        preprocess_button.pack(pady=10)

        # 训练按钮
        train_button = tk.Button(self.root, text="训练模型", command=self.train_model)
        train_button.pack(pady=10)

        # 模型选择（预测）
        self.predict_model_var = tk.StringVar(value="LSTM")
        predict_model_label = tk.Label(self.root, text="选择预测模型:", font=("Arial", 14))
        predict_model_label.pack(pady=5)
        # predict_model_menu = ttk.Combobox(self.root, textvariable=self.predict_model_var, values=["LSTM", "CNN", "GRU", "Transformer"])
        predict_model_menu = ttk.Combobox(self.root, textvariable=self.predict_model_var,
                                          values=["LSTM", "GRU", "Transformer"])
        predict_model_menu.pack(pady=5)

        # 上传文件按钮
        upload_button = tk.Button(self.root, text="上传语音文件", command=self.upload_file)
        upload_button.pack(pady=10)

        # 显示 MFCC 图像
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(pady=10)

        # 结果显示
        self.result_label = tk.Label(self.root, text="预测结果: ", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def preprocess_data(self):
        # 预处理数据并保存
        preprocess_and_save(self.data_dir, self.preprocessed_dir)
        messagebox.showinfo("预处理完成", "数据预处理完成并已保存！")

    def train_model(self):
        # 加载预处理后的数据集
        dataset = PreprocessedDataset(self.preprocessed_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        # 初始化模型
        model_type = self.train_model_var.get()
        if model_type == "LSTM":
            model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.num_classes)
        # elif model_type == "CNN":
        #     model = CNNModel(self.num_classes)
        elif model_type == "GRU":
            model = GRUModel(self.input_size, self.hidden_size, self.num_layers, self.num_classes)
        elif model_type == "Transformer":
            model = TransformerModel(self.input_size, self.num_heads, self.num_layers, self.num_classes)
        else:
            raise ValueError("未知模型类型")

        # 训练模型
        model = train_model(model, train_loader, val_loader)

        # 保存模型
        model_path = f"{model_type.lower()}_model.pth"
        torch.save(model.state_dict(), model_path)
        messagebox.showinfo("训练完成", f"{model_type} 模型训练完成并已保存为 {model_path}！")

    def upload_file(self):
        model_type = self.predict_model_var.get()
        model_path = f"{model_type.lower()}_model.pth"

        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"未找到 {model_type} 模型文件，请先训练模型！")
            return

        # 加载模型
        if model_type == "LSTM":
            model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.num_classes)
        # elif model_type == "CNN":
        #     model = CNNModel(self.num_classes)
        elif model_type == "GRU":
            model = GRUModel(self.input_size, self.hidden_size, self.num_layers, self.num_classes)
        elif model_type == "Transformer":
            model = TransformerModel(self.input_size, self.num_heads, self.num_layers, self.num_classes)
        else:
            raise ValueError("未知模型类型")

        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        # 上传文件
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            # 提取特征
            waveform, _ = torchaudio.load(file_path)
            mfcc = extract_mfcc(waveform)

            # 显示 MFCC 图像
            self.ax.clear()
            self.ax.imshow(mfcc.T, origin="lower", aspect="auto")
            self.ax.set_title("MFCC Features")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("MFCC Coefficients")
            self.canvas.draw()

            # 预测情感
            features = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)  # 将数据移动到 GPU
            with torch.no_grad():
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                emotion = list(emotion_map.keys())[predicted.item()]
                self.result_label.config(text=f"预测结果: {emotion}")

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()