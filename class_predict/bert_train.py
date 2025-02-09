import torch
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# 1. 读取数据
train_data = pd.read_csv('./train.csv', sep='\t')

# 查看数据的前几行，确保数据正确加载
print(train_data.head())

# 2. 中文分词函数
def jieba_tokenizer(text):
    return list(jieba.cut(text))

# 3. 数据预处理
# 对每个句子进行分词
train_data['Phrase'] = train_data['Phrase'].apply(lambda x: ' '.join(jieba_tokenizer(x)))

# 情感类别映射
sentiment_to_idx = {
    "恐惧": 0,
    "愤怒": 1,
    "厌恶": 2,
    "开心": 3,
    "伤心": 4,
    "惊讶": 5
}

# 将情感标签转换为数字
train_data['Sentiment'] = train_data['Sentiment'].map(sentiment_to_idx)

# 4. 划分数据集
X = train_data['Phrase']
y = train_data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. 使用BERT Tokenizer进行编码
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def encode_data(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

train_encodings = encode_data(X_train)
test_encodings = encode_data(X_test)

# 转换为PyTorch tensor
train_labels = torch.tensor(y_train.values, dtype=torch.long)
test_labels = torch.tensor(y_test.values, dtype=torch.long)

# 6. 创建DataLoader
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # 7. BERT模型定义
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(sentiment_to_idx))

# # 8. 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# 9. 训练过程
epochs = 3  # 根据需要调整epochs
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    torch.save(model.state_dict(), 'bert_finetuned_sentiment.pth')

model = model.to(device)  # 移动模型到GPU或CPU

# 加载模型的状态字典
model.load_state_dict(torch.load('bert_finetuned_sentiment.pth'))
# 10. 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# # 11. 预测情感
# def predict_sentiment(text, model, tokenizer, sentiment_to_idx):
#     model.eval()  # 设置为评估模式
#     text_cut = ' '.join(jieba_tokenizer(text))  # 分词并拼接成空格分隔的字符串
#     inputs = tokenizer(text_cut, padding=True, truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)

#     with torch.no_grad():
#         output = model(input_ids, attention_mask=attention_mask)
#         logits = output.logits
#         _, predicted = torch.max(logits, 1)
        
#         # 获取反向映射
#         idx_to_sentiment = {v: k for k, v in sentiment_to_idx.items()}
#         predicted_label = idx_to_sentiment[predicted.item()]
        
#         return predicted_label


# 12. 验证外部数据集
# 假设您的外部数据集路径为'./external_data.csv'
external_data = pd.read_csv('./test.csv', sep='\t')

# 确保外部数据集格式正确
print(external_data.head())

# 中文分词
external_data['Phrase'] = external_data['Phrase'].apply(lambda x: ' '.join(jieba_tokenizer(x)))

# 将情感标签转换为数字
external_data['Sentiment'] = external_data['Sentiment'].map(sentiment_to_idx)

# 使用训练时的BERT Tokenizer转换外部数据集
external_encodings = encode_data(external_data['Phrase'])

# 转换为PyTorch tensor
external_labels_tensor = torch.tensor(external_data['Sentiment'].values, dtype=torch.long)

# 创建DataLoader
external_dataset = TensorDataset(external_encodings['input_ids'], external_encodings['attention_mask'], external_labels_tensor)
external_loader = DataLoader(external_dataset, batch_size=32, shuffle=False)

# 加载模型
model.load_state_dict(torch.load('bert_finetuned_sentiment.pth'))
model = model.to(device)  # 确保模型在正确的设备上

# 测试外部数据集
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(external_loader, desc="External Testing"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

external_accuracy = 100 * correct / total
print(f"External Test Accuracy: {external_accuracy:.2f}%")