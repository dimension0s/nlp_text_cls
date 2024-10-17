# 2.构建模型：fasttext快速文本分类模型
# 注意：该模型结构过于简单，容易过拟合，如果要修改，可以考虑增加维度

import torch.nn.functional as F
import torch.nn as nn
from text_cls.collate_fn import tokenizer
from text_cls.device import device

class Fasttext(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_classes):
        super().__init__()
        # 词向量嵌入
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        # 权重衰减
        self.dropout = nn.Dropout(0.5)
        self.cls1 = nn.Linear(embed_dim,hidden_size)
        self.cls2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        embedded = self.embedding(x['input_ids'])
        embedded = embedded.mean(dim=1)
        embedded = self.dropout(embedded)
        output = self.cls1(embedded)
        output = F.relu(output)
        output = self.dropout(output)
        logits = self.cls2(output)
        return logits

vocab_size = tokenizer.vocab_size
model = Fasttext(vocab_size,embed_dim=128,hidden_size=72,num_classes=10)
model = model.to(device)
print(model)