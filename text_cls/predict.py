# 5.预测函数封装
from text_cls.device import device
import torch
from text_cls.models.fasttext import model,Fasttext,vocab_size
from text_cls.data import test_data
from text_cls.collate_fn import tokenizer
def predicted(text,model,tokenizer):
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text,padding=True,truncation=True,
                    return_tensors="pt").to(device)
        logits = model(inputs)
        predictions = torch.argmax(logits,dim=1)
    return predictions

# 建立数字到文字标签的映射字典
label_mapping={
    0:'finance',
    1:'realty',
    2:'stocks',
    3:'education',
    4:'science',
    5:'society',
    6:'politics',
    7:'sports',
    8:'game',
    9:'entertainment'
}

# 将预测标签的数字转换为文本标签

model = Fasttext(vocab_size,embed_dim=128,hidden_size=72,num_classes=10)
# 预测函数示例

for i in range(15):
    text=test_data[i]['comment']
    prediction=predicted(text,model,tokenizer)
    predicted_label_text=label_mapping[prediction.item()]
    print(f'Text:{text}')
    print(f'Prediction:{prediction.item()}')
    print('='*50)