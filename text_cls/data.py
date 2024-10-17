# 1.构建数据集
# 1.1）加载数据集

from torch.utils.data import Dataset,DataLoader

class THUCNews(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file,'rt',encoding='utf-8') as f:
            for idx,line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2
                Data[idx] = {
                    'comment': items[0],
                    'label': items[1],
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = THUCNews("E:\\NLPProject\\text_cls\\data\\train.txt")
valid_data = THUCNews("E:\\NLPProject\\text_cls\\data\\dev.txt")
test_data = THUCNews("E:\\NLPProject\\text_cls\\data\\test.txt")

# 输出数据集尺寸，打印一个训练样本
print(f'train set size:{len(train_data)}')
print(f'valid set size:{len(valid_data)}')
print(f'test set size:{len(test_data)}')
print(next(iter(train_data)))