# 2.构建模型：TextRNN模型：捕捉长距离语义关系
# 架构说明：词向量（embedding）---Bi-LSTM(/GRU) ---取最后一个时间步隐状态并拼接
# （或：取每个时间步隐状态再拼接，再取平均）---FC（线性全连接层）---softmax分类
# 插入步骤：1.优化与正则化（防止过拟合），比如L2，dropout
# 2.BatchNormalization:层规范化，加速模型训练
import torch.nn as nn
from text_cls.collate_fn import tokenizer
from text_cls.device import device


# 模型1
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x['input_ids'])  # [batch_size,seq_len,embedding]=[128,32,300]
        out, _ = self.lstm(out)  # 输出隐状态和细胞元状态，后者被忽视
        out = self.fc(out[:, -1, :])  # 取最后时间步的隐状态 hidden state
        return out


vocab_size = tokenizer.vocab_size
embed_dim = 154
hidden_size = 128
num_layers = 2
num_classes = 10
model_rnn = TextRNN(vocab_size, embed_dim, hidden_size, num_layers, num_classes)
print(model_rnn)