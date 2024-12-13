import os
import numpy as np
import pandas as pd
import torch.nn as nn
import pandas as pd
import torch
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
from sklearn.model_selection import ParameterGrid


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

file_path = "/home/develop/GATv2/data/final.csv"
data_new = pd.read_csv(file_path)

features_new = torch.tensor(data_new.values, dtype=torch.float32).T  # 转置，维度 (42, N)
num_features_new = features_new.size(0)  # 42


adj_matrix_new = torch.ones((num_features_new, num_features_new)) - torch.eye(num_features_new)  # 完全图
edge_index_new = dense_to_sparse(adj_matrix_new)[0]

# 4. 创建图数据对象
device = torch.device('cpu')  # 使用 CPU
graph_data_new = Data(x=features_new, edge_index=edge_index_new)

# 5. 加载训练好的 GATv2 模型
model = torch.load("/home/develop/GATv2/Model/GATv2_trained.pth")
model.eval()  # 设置为评估模式

# 6. 确保数据移到正确的设备
graph_data_new = graph_data_new.to(device)

# 7. 推理
with torch.no_grad():  # 禁用梯度计算
    output_new = model(graph_data_new.x, graph_data_new.edge_index)  # 获得模型输出

output_new_transposed = output_new.T  # 转置后变为 [1186, 42]
output_new_transposed_df = pd.DataFrame(output_new_transposed.numpy())


# 1. 数据预处理
X = output_new_transposed_df.values  # 转为numpy数组

# 划分训练集和验证集，按8:2划分
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# 定义采样层
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# 定义编码器部分为独立的模型
class Encoder(layers.Layer):
    def __init__(self, latent_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(64, activation="relu")
        self.dense_2 = layers.Dense(32, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling_layer = layers.Lambda(sampling)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z

# 定义解码器部分为独立的模型
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

# 定义VAE模型
class VAE(Model):
    def __init__(self, original_dim, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(original_dim=original_dim)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = tf.reduce_mean(reconstruction_loss_fn(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
            )
            total_loss = reconstruction_loss + 0.1 * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # 注意：这里data直接是输入特征，而不是元组 (input, target)
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = tf.reduce_mean(reconstruction_loss_fn(data, reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        total_loss = reconstruction_loss + 0.1 * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

# 网格搜索训练过程
with open(log_file, "w") as log:
    log.write("Training Log\n")
    log.write("Parameters: latent_dim, batch_size, epochs\n")
    log.write("Results: reconstruction_error_threshold, anomalies_detected, training_time\n")
    log.write("-" * 80 + "\n")

    for latent_dim in latent_dims:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                input_dim = X_train.shape[1]

                # 构建VAE模型
                vae = VAE(original_dim=input_dim, latent_dim=latent_dim)
                vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))

                # 训练模型
                start_time = datetime.now()
                history = vae.fit(
                    X_train, 
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test,),  # 只传递验证数据，不包括目标
                    verbose=0
                )
                training_time = datetime.now() - start_time

                # 绘制训练损失
                plt.figure(figsize=(10, 6))
                
                # 检查是否存在验证损失键
                if 'val_loss' in history.history:
                    plt.plot(history.history["loss"], label="Train Loss")
                    plt.plot(history.history["val_loss"], label="Validation Loss")
                else:
                    plt.plot(history.history["loss"], label="Loss")
                
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.title(f"Loss (latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs})")
                plt.savefig(f"{save_dir}loss_latent{latent_dim}_batch{batch_size}_epochs{epochs}.png")
                plt.close()

                # 异常检测
                X_pred = vae.predict(X_test, verbose=0)
                reconstruction_error = np.mean(np.square(X_test - X_pred), axis=1)
                threshold = np.percentile(reconstruction_error, 95)
                anomalies = reconstruction_error > threshold

                # 绘制重构误差分布
                plt.figure()
                plt.hist(reconstruction_error, bins=50)
                plt.xlabel("Reconstruction Error")
                plt.ylabel("Number of Samples")
                plt.title(f"Error Dist. (latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs})")
                plt.savefig(f"{save_dir}error_dist_latent{latent_dim}_batch{batch_size}_epochs{epochs}.png")
                plt.close()

                # 记录日志
                log.write(f"latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs}\n")
                log.write(f"reconstruction_error_threshold={threshold:.4f}, anomalies_detected={np.sum(anomalies)}, training_time={training_time}\n")
                log.write("-" * 80 + "\n")
                print(f"Params: latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs}")
                print(f"Reconstruction Error Threshold: {threshold:.4f}")
                print(f"Anomalies detected: {np.sum(anomalies)}")
                print(f"Training Time: {training_time}\n")