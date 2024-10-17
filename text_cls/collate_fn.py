# 1.2)对数据集分批，分词和编码
# 分批函数
import torch
from transformers import AutoTokenizer,AutoConfig
from text_cls.data import train_data,valid_data,test_data
from torch.utils.data import DataLoader

checkpoint = 'bert-base-chinese'  # bert-base-chinese是针对中文的预训练语言模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples):
    batch_comment, batch_label = [],[]
    for sample in batch_samples:
        batch_comment.append(sample['comment'])
        batch_label.append(int(sample['label']))

    batch_data = tokenizer(
        batch_comment,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    labels = torch.tensor(batch_label)

    return batch_data,labels

train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=64,shuffle=True,collate_fn=collote_fn)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True,collate_fn=collote_fn)

# 打印一批数据集
batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:',{k:v.shape for k,v in batch_X.items()})
print('batch_y shape:',batch_y.shape)
print(batch_X)
print(batch_y)

# 对以上做个总结：
# 1.使用的是transformers中的自动分词器tokenizer，包含词表大小
# 2.在原始做法中，建立词表的目的是为分词之后的编码，给词元分配索引