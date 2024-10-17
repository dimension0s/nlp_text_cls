# 3.模型训练与验证，此处相比原来版本有很大修改：
# 3.1) 训练函数:增加准确率correct和进度条，在epoch中加入了可视化

import os,torch
from text_cls.device import device
from tqdm.auto import tqdm

def train_loop(dataloader,model,loss_fn,optimizer,scheduler,epoch):
    total_loss = 0.
    correct = 0.
    total = 0

    model.train()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 计算训练集的准确率
        _,predicted = torch.max(pred, 1)
        # 最后算得是平均准确率
        correct += (predicted == y).sum().item()
        total += y.size(0)

        avg_loss = total_loss / (step + 1)
        accuracy = correct/total
        progress_bar.set_description(f'Epoch {epoch},Loss:{avg_loss:.4f},Accuaracy:{accuracy:.4f}')
    return avg_loss, accuracy



