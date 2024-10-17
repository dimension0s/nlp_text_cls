# textCNN模型构建用于文本分类任务

# 2.构建模型：TextCNN模型：捕捉单词之间的关系
# 架构说明：词向量（word2vec,glove,embedding）---卷积与池化---全连接---softmax分类
import torch.nn.functional as F
from torch import nn
from text_cls.collate_fn import tokenizer
import torch

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 创建多个卷积层
        # 参数说明：
        # 1.vocab_size：输入通道数，对应文本维度
        # num_filters:输出通道数，决定卷积核数量，也是滤波器数量，即提取的特征数量
        # (k,embed_dim)：卷积核大小：
        # k：卷积核高度，embed_dim：嵌入词向量维度，也就是卷积核宽度
        # filter_sizes：不同卷积核尺寸，决定了模型能够捕捉的不同n-gram特征：
        # 比如：3-gram 4-gram 5-gram
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        # conv(x)后得到的维度：(batch_size, num_filters,seq_len,1)
        # 最后一个维度没用，去掉
        x = F.relu(conv(x).squeeze(3))
        # x.size(2)：对经过ReLU激活的特征图进行最大池化操作
        # 这里池化操作在序列长度的维度上进行，
        # 将每个特征图的每个通道在序列长度维度上进行最大池化。
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # 初始维度：(batch_size, seq_len)
        # step1.维度：(batch_size, seq_len, embed_dim)
        out = self.embedding(x['input_ids'])
        # 在进行卷积操作之前，我们需要将词嵌入的维度扩展一维，以适应卷积操作的输入要求。
        # 使用.unsqueeze(1) 将维度扩展为 (batch_size, 1, seq_len, embed_dim)
        out = out.unsqueeze(1)
        # 对每个卷积层应用 conv_and_pool 方法,并将它们的输出在通道维度上进行拼接
        # 这样做是为了获得不同大小卷积核的特征的组合.
        # step3.维度：(batch_size, num_filters)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        # step4.维度变化：(batch_size, num_classes)
        out = self.fc(out)
        return out


vocab_size = tokenizer.vocab_size
embed_dim = 128
num_filters = 96  # 卷积核数量，channels数,，通道数，特征数
filter_sizes = (2, 3, 4)  # 卷积核大小，滤波器
num_classes = 10
model = TextCNN(vocab_size, embed_dim, num_filters, filter_sizes, num_classes)
print(model)