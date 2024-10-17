# 3.2) 验证/测试函数
# 在测试函数中，通常不需要计算训练集损失，
# 因为测试函数主要评估模型在验证集和测试集上的性能。

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from text_cls.device import device
import torch
import numpy

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    all_predictions, all_labels = [], []

    model.eval()
    model = model.to(device)
    total_loss = 0.
    correct = 0.
    total = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred_probs = model(X)
            _, preds = torch.max(pred_probs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 计算准确率
    accuracy = correct / total

    # 计算精确率，召回率，F1,等评价指标
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1