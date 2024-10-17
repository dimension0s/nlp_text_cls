# 4.主训练循环

from transformers import AdamW,get_linear_schedule_with_warmup
import random,os
import numpy as np
import torch
from text_cls.device import device
import torch.nn as nn
from text_cls.collate_fn import train_dataloader,valid_dataloader
from text_cls.models.fasttext import model
from text_cls.train import train_loop
from text_cls.test import test_loop

def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

learning_rate = 0.0003
epoch_num = 45

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader))
best_accuracy=0.
best_valid_loss=float('inf')
for epoch in range(epoch_num):
    print(f'Epoch {epoch + 1}/{epoch_num}\n-------------------------------')
    # 训练模型
    train_loss, train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer,
                                            scheduler, epoch + 1)

    # 验证模型
    valid_accuracy,valid_precision, valid_recall, valid_f1 = test_loop(
        valid_dataloader, model, mode='Valid')

    # 保存最佳模型权重
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        print('Saving new weights......')
        torch.save(
            model.state_dict(),
            f'epoch_{epoch + 1}_valid_accuracy_{valid_accuracy:.4f}_weights.pth')

        # 打印验证集的评价指标
        print(f'Validation Accuracy: {valid_accuracy:.4f}, '
              f'Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1: {valid_f1:.4f}')

