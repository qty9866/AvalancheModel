import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# GATv2 模型定义
class GATv2Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GATv2Net, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    
    # 移除 yichang 列，如果存在的话
    if 'yichang' in data.columns:
        features = data.drop(columns=['yichang']).values
    else:
        features = data.values
    
    features = torch.tensor(features, dtype=torch.float32).T 
    num_features = features.size(0)
    
    adj_matrix = torch.ones((num_features, num_features)) - torch.eye(num_features)  # 完全图
    edge_index = dense_to_sparse(adj_matrix)[0]
    
    return Data(x=features, edge_index=edge_index)

def inference(model, file_path):
    model.eval()  # 设置为评估模式
    graph_data_new = prepare_data(file_path).to(device)

    with torch.no_grad():
        output_new = model(graph_data_new.x, graph_data_new.edge_index)  # 获得模型输出

    # 打印输出
    print("GATv2 模型的输出：")
    print(output_new)  # 打印推理结果
    print(f"模型输出的维度: {output_new.shape}")
    return output_new.T.numpy()  # 返回转置后的numpy数组

# VAE相关部分
def build_vae(input_dim, latent_dim):
    class Encoder(layers.Layer):
        def __init__(self, latent_dim, **kwargs):
            super(Encoder, self).__init__(**kwargs)
            self.dense_1 = layers.Dense(64, activation="relu")
            self.dense_2 = layers.Dense(32, activation="relu")
            self.dense_mean = layers.Dense(latent_dim)
            self.dense_log_var = layers.Dense(latent_dim)

        def call(self, inputs):
            x = self.dense_1(inputs)
            x = self.dense_2(x)
            z_mean = self.dense_mean(x)
            z_log_var = self.dense_log_var(x)
            return z_mean, z_log_var

    class Decoder(layers.Layer):
        def __init__(self, original_dim, **kwargs):
            super(Decoder, self).__init__(**kwargs)
            self.dense_1 = layers.Dense(32, activation="relu")
            self.dense_2 = layers.Dense(64, activation="relu")
            self.dense_output = layers.Dense(original_dim, activation="sigmoid")

        def call(self, inputs):
            x = self.dense_1(inputs)
            x = self.dense_2(x)
            return self.dense_output(x)

    class VAE(Model):
        def __init__(self, original_dim, latent_dim, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.original_dim = original_dim
            self.encoder = Encoder(latent_dim=latent_dim)
            self.decoder = Decoder(original_dim=original_dim)

        def call(self, inputs):
            z_mean, z_log_var = self.encoder(inputs)
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            return self.decoder(z)

    vae = VAE(original_dim=input_dim, latent_dim=latent_dim)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    return vae

# 超参数
hidden_channels = 8  # 隐藏层维度
heads = 4  # 多头注意力

# 数据路径
data_file_path = '/home/develop/GATv2-VAE_NewData/data/csv/test_temp.csv'

# 设备配置
device = torch.device('cpu')  # 强制使用 CPU

# 初始化模型（这里假设模型的输入和输出维度与训练时相同）
graph_data = prepare_data(data_file_path)
in_channels = graph_data.x.size(1)
out_channels = graph_data.x.size(1)  # 输出维度等于输入维度（重构任务）

model = GATv2Net(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,  # 输出维度等于输入维度（重构任务）
    heads=heads
).to(device)

# 使用模型进行推理
output_new_transposed = inference(model, data_file_path)

# 将GATv2的输出转换为DataFrame
output_new_transposed_df = pd.DataFrame(output_new_transposed)

# 构建VAE模型
latent_dim = 20
input_dim = output_new_transposed_df.shape[1]

vae = build_vae(input_dim, latent_dim)

# 直接对整个数据集进行推理
X_pred = vae.predict(output_new_transposed_df.values, batch_size=16, verbose=0)
reconstruction_error = np.mean(np.square(output_new_transposed_df.values - X_pred), axis=1)

# 根据87百分位数检测异常
threshold = np.percentile(reconstruction_error, 80)
anomalies = reconstruction_error > threshold

# 输出异常数据的索引
anomalous_indices = np.where(anomalies)[0]
print("Anomalous indices:")
print(anomalous_indices)
# 数据路径
data_file_path = '/home/develop/GATv2-VAE_NewData/data/csv/test_temp.csv'

# 加载数据
data = pd.read_csv(data_file_path)

# 确认 'yichang' 列存在
if 'yichang' not in data.columns:
    print("No yichang column found.")
else:
    # 获取指定索引对应的 'yichang' 值
    anomalous_origin_indices = data.loc[anomalous_indices, 'yichang'].values
    
    # 打印结果
    print("Origin indices for the specified rows:")
    print(anomalous_origin_indices)