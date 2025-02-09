import torch
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
import os
os.environ["HUGGINGFACE_HUB_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-hub"
os.environ["HF_DATASETS_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-datasets"
os.environ["HF_METRICS_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-metrics"
os.environ["HF_HOME"] = "/path/to/your/hf_home_directory"
os.environ["TRANSFORMERS_CACHE"] = "./class_preidct"
os.environ["HF_DATASETS_CACHE"] = "./class_preidct"


sentiment_to_idx = {
    "恐惧": 0,
    "愤怒": 1,
    "厌恶": 2,
    "开心": 3,
    "伤心": 4,
    "惊讶": 5
}


def jieba_tokenizer(text):
    return list(jieba.cut(text))

def predict_sentiment(text, model, tokenizer, sentiment_to_idx):
    model.eval()  # 设置为评估模式
    text_cut = ' '.join(jieba_tokenizer(text))  # 分词并拼接成空格分隔的字符串
    inputs = tokenizer(text_cut, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        _, predicted = torch.max(logits, 1)
        
        # 获取反向映射
        idx_to_sentiment = {v: k for k, v in sentiment_to_idx.items()}
        predicted_label = idx_to_sentiment[predicted.item()]
        
        return predicted_label
    

local_model_path = os.path.abspath('./class_predict/local_bert_base_chinese')


model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=len(sentiment_to_idx))
model.load_state_dict(torch.load('./class_predict/bert_finetuned_sentiment.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(local_model_path)
text = input()
print(predict_sentiment(text,model,tokenizer,sentiment_to_idx))

