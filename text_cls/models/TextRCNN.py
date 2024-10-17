# 构建模型：TextRCNN模型：学习更多的上下文信息，并且更准确地表示文本语义
# 该模型结合了RNN的结构和最大池化层，利用了循环神经模型和卷积神经模型的优点
# 架构说明：

import torch.nn as nn
import torch.nn.functional as F
from text_cls.device import device
from text_cls.collate_fn import tokenizer
import torch


class TextRCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 conv_out_channels, kernel_size, pad_size,
                 num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=0.5)

        # 添加卷积层
        self.conv = nn.Conv1d(hidden_size * 2, out_channels=conv_out_channels,
                              kernel_size=kernel_size)

        # 修改最大池化层以匹配卷积后的输出
        # self.maxpool = nn.MaxPool1d(kernel_size=pad_size)
        # 将最大池化改成全局最大池化，可以解决后期匹配问题，因为确保了输出宽度为1
        # 也可以使用平均池化nn.AdaptiveAvgPool1d(1)，效果是一样的
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 拼接操作：将卷积层的输出和LSTM的隐藏状态拼接在一起
        # 注意，我们需要确保LSTM的输出和卷积层输出的维度匹配
        self.fc = nn.Linear(hidden_size * 2 + conv_out_channels, num_classes)

    def forward(self, x):
        embed = self.embedding(x['input_ids'])
        lstm_out, _ = self.lstm(embed)  # [batch_size, seq_len, embedding_dim]

        # LSTM输出维度: [batch_size, seq_len, hidden_size * 2]
        # 将LSTM输出转换为适合卷积层的格式: [batch_size, hidden_size * 2, seq_len]
        lstm_out = lstm_out.permute(0, 2, 1)

        # 应用卷积层
        # 输入维度：(batch_size, hidden_size*2, sequence_length)
        # 输出维度：(batch_size, conv_out_channels, sequence_length - kernel_size + 1)
        # 其中：sequence_length - kernel_size + 1=conv_seq_len=25-3+1=23，此处没有使用padding
        # 因此输出维度还可以写成：
        # (batch_size, conv_out_channels,conv_seq_len)
        conv_out = self.conv(lstm_out)

        # 应用ReLU激活函数，维度不变
        conv_out = F.relu(conv_out)

        # 应用最大池化层
        # 注意，池化层的输出维度可能会变化，取决于输入序列长度和pad_size
        # 输入维度：(batch_size, conv_out_channels, sequence_length - kernel_size + 1)
        # 输出维度：(batch_size, conv_out_channels, conv_seq_len//pad_size)
        # 其中对于conv_seq_len//pad_size向下取整，因此该维度变成0：23//16=0
        # pooled_out = self.maxpool(conv_out)

        # 更换池化后，输出维度变成：(batch_size, conv_out_channels, 1)，
        # 此处操作解决了后期维度不匹配的问题。
        pooled_out = self.pool(conv_out).squeeze(-1)

        # 将LSTM的最后一个隐藏状态和卷积层的输出拼接
        # LSTM的最后一个隐藏状态维度: [batch_size, hidden_size * 2]
        # 提取后维度：[batch_size, hidden_size * 2,1]
        last_hidden = lstm_out[:, :, -1]
        #         print('last_hidden shape:',last_hidden.shape)

        # 拼接LSTM的最后一个隐藏状态和卷积层的输出
        combined = torch.cat((last_hidden, pooled_out), dim=-1)

        # 通过全连接层得到最终输出
        out = self.fc(combined)

        return out


vocab_size = tokenizer.vocab_size
embed_dim = 154
# 一般要比embed_dim更大，或者词嵌入的维度的倍数，以便模型能够学习更复杂的特征
hidden_size = 256
num_layers = 2
conv_out_channels = 128  # 通常在100-500之间
kernel_size = 3  # 通常选择小于输入序列长度的值，例如 3 或 5，以捕捉局部特征
# 注意：要确保pad_size 不大于输入序列的最大长度减去卷积核的大小
pad_size = 16  # 可以根据输入序列的最大长度来选择，通常设置为一个较小的固定值
num_classes = 10

model = TextRCNN(vocab_size, embed_dim, hidden_size, num_layers, conv_out_channels,
                 kernel_size, pad_size, num_classes)
print(model)