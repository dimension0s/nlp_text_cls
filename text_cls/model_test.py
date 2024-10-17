# 5.模型测试
from text_cls.data import test_data
from text_cls.collate_fn import test_dataloader
import json,torch
from text_cls.models.fasttext import model
from text_cls.device import device
from text_cls.test import test_loop

model.load_state_dict(torch.load('epoch_57_valid_accuracy_0.8717_weights.pth'))
model.eval()
model = model.to(device)
# 结果记录与输出
results = []
print('evaluating on test set...')
test_avg_loss, accuracy, precision, recall, test_f1 = test_loop(test_dataloader, model, mode='Test')
results.append(
    {
        'test_avg_loss': test_avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'test_f1': test_f1,

    })
print(results)


