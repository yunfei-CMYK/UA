import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn
from TorchCRF import CRF
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import datetime

# 配置日志记录
log_dir = '../log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# 获取当前日期
current_date = datetime.datetime.now().strftime("%Y%m%d")
# 获取程序文件名
script_name = os.path.basename(__file__)
log_file = os.path.join(log_dir, f"{current_date}_{script_name.replace('.py', '.log')}")

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 定义焦点损失函数类
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, inputs, targets, mask):
        """
        inputs: [batch_size, seq_len, num_labels]
        targets: [batch_size, seq_len]
        mask: [batch_size, seq_len] (bool mask for valid tokens)
        """
        # Flatten the inputs and targets
        inputs_flat = inputs.view(-1, inputs.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Cross-Entropy Loss
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs_flat, targets_flat)
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets_flat)
            F_loss = at * F_loss

        # Apply mask to ignore padding tokens
        F_loss = F_loss * mask_flat.float()

        return F_loss.sum() / mask_flat.sum()

# 读取 BIO 格式的 txt 数据
def read_data_from_txt(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        current_sentence = []
        current_labels = []
        for idx, line in enumerate(lines):
            line = line.strip()
            if line == "":  # 遇到空行，代表一句话结束
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                try:
                    word, label = line.split('\t')
                    current_sentence.append(word)
                    current_labels.append(label)
                except ValueError:
                    logging.error(f"Error parsing line {idx + 1}: {line}")
                    continue  # 如果分割不正确，跳过当前行
        if current_sentence:  # 如果文件结尾没有空行
            sentences.append(current_sentence)
            labels.append(current_labels)
    return sentences, labels

# NER数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {
            'O': 0, 'B-动作': 1, 'I-动作': 2, 'B-组件': 3, 'I-组件': 4,
            'B-传感器': 5, 'I-传感器': 6, 'B-方法': 7, 'I-方法': 8, 'B-位置': 9, 'I-位置': 10,
            'B-参数': 11, 'I-参数': 12, 'B-故障类型': 13, 'I-故障类型': 14, 'B-材料': 15, 'I-材料': 16,
            'B-工具': 17, 'I-工具': 18, 'B-特点': 19, 'I-特点': 20, 'B-人员': 21, 'I-人员': 22,
            'B-无人机类型': 23, 'I-无人机类型': 24, 'B-领域': 25, 'I-领域': 26, 'B-术语': 27, 'I-术语': 28,
            'B-影响因素': 29, 'I-影响因素': 30, 'B-功能': 31, 'I-功能': 32
        }

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=False,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # 将标签对齐到BERT的tokenizer输出
        labels = [self.label_map.get(l, 0) for l in label[:self.max_len]]
        labels = labels + [0] * (self.max_len - len(labels))

        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels)
        return item

# 构建BERT+BiLSTM+CRF模型
class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, num_labels, hidden_dim=256, gamma=2, alpha=None):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)  # 初始化焦点损失函数

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取BERT输出
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # BiLSTM层
        lstm_output, _ = self.lstm(sequence_output)

        # 线性层
        emissions = self.fc(lstm_output)

        if labels is not None:
            # CRF层 (训练模式)
            crf_loss = -self.crf(emissions, labels, mask=attention_mask.bool())  # 使用 .bool()
            crf_loss = crf_loss.mean()  # 返回标量

            # 使用焦点损失函数计算损失
            focal_loss = self.focal_loss(emissions, labels, attention_mask)
            total_loss = crf_loss + focal_loss

            return total_loss
        else:
            # CRF层 (推理模式)
            prediction = self.crf.viterbi_decode(emissions, mask=attention_mask.bool())  # 使用 .bool()
            return prediction

# 训练模型，只在最后一次epoch后保存模型
def train_model(model, dataloader, optimizer, epochs=100, batch_size=16, model_save_path='./saved_model',
                save_model=True, save_log=True):
    model.train()

    if save_log:
        # 记录表头到日志
        logging.info("Epoch, Loss, Accuracy, Precision, Recall, F1")

    for epoch in range(epochs):
        total_loss = 0
        all_labels = []
        all_predictions = []

        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 获取模型预测结果，用于评估指标
            with torch.no_grad():
                preds = model(input_ids, attention_mask)  # 返回预测的标签
                for pred, label, mask in zip(preds, labels, attention_mask):
                    valid_length = mask.sum().item()  # 获取有效长度，去除padding
                    all_predictions.extend(pred[:valid_length])  # 添加有效长度的预测结果
                    all_labels.extend(label[:valid_length].cpu().numpy())  # 添加有效长度的真实标签

        # 计算损失
        avg_loss = total_loss / len(dataloader)

        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # 打印每个 epoch 的评估结果
        if save_log:
            logging.info(f"{epoch + 1}, {avg_loss}, {accuracy}, {precision}, {recall}, {f1}")

    # 所有epochs完成后保存模型
    if save_model:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        final_model_path = os.path.join(model_save_path, 'bert_bilstm_crf_focal_loss.pt')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Model saved after final epoch at {final_model_path}")

# 加载保存的模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 测试完整句子的识别
def predict_sentence(model, tokenizer, sentence, max_len=256):
    model.eval()
    # 将输入句子拆分成字符或词语
    sentence_tokens = list(sentence)

    # 对输入句子进行tokenize
    encoding = tokenizer(sentence_tokens,
                         is_split_into_words=True,
                         return_offsets_mapping=False,
                         padding='max_length',
                         truncation=True,
                         max_length=max_len,
                         return_tensors='pt')

    input_ids = encoding['input_ids'].to('cuda')
    attention_mask = encoding['attention_mask'].to('cuda')

    with torch.no_grad():
        preds = model(input_ids, attention_mask)

    # 将预测结果转换为标签
    pred_labels = preds[0]  # 只取第一个样本（单句）

    # 将标签索引映射回标签名
    idx2tag = {v: k for k, v in dataset.label_map.items()}
    pred_tags = [idx2tag.get(p, 'O') for p in pred_labels]

    # 过滤掉padding部分
    valid_pred_tags = pred_tags[:attention_mask.sum().item()]

    return list(zip(sentence_tokens, valid_pred_tags))

# 在测试集上评估模型性能
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=True):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            preds = model(input_ids, attention_mask)

            for pred, label, mask in zip(preds, labels, attention_mask):
                valid_length = mask.sum().item()  # 获取有效长度，去除padding
                all_predictions.extend(pred[:valid_length])  # 添加有效长度的预测结果
                all_labels.extend(label[:valid_length].cpu().numpy())  # 添加有效长度的真实标签

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    logging.info(f'Test Set Evaluation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

# 主程序
if __name__ == "__main__":
    # 参数配置
    data_file = '../datasets/bio_data/all_bio_data.txt'  # 替换为你的数据文件路径
    bert_model_name = 'bert-base-chinese'
    batch_size = 32
    hidden_dim = 256
    max_len = 256
    model_save_path = './saved_model'

    test_sentence = '民用无人机大范围使用，其中包含了大量的传感器，比如姿态传感器。'

    model_epochs = 100  # 统一配置训练次数

    # 添加是否保存模型和日志的开关
    save_model_flag = True
    save_log_flag = True

    # 加载数据集
    sentences, labels = read_data_from_txt(data_file)

    # 使用BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    # 计算数据集的总大小
    total_size = len(sentences)

    # 按照 8:1:1 划分训练集、验证集和测试集
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # 创建NER数据集
    dataset = NERDataset(sentences, labels, tokenizer)

    # 使用 random_split 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 打印每个数据集的大小
    logging.info(f'Train dataset size: {len(train_dataset)}')
    logging.info(f'Validation dataset size: {len(val_dataset)}')
    logging.info(f'Test dataset size: {len(test_dataset)}')

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = BERT_BiLSTM_CRF(bert_model_name, num_labels=len(dataset.label_map), hidden_dim=hidden_dim)
    model = model.to('cuda')

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 检查模型是否已存在
    final_model_path = os.path.join(model_save_path, 'bert_bilstm_crf_focal_loss.pt')
    if os.path.exists(final_model_path):
        logging.info("Loading existing model...")
        model = load_model(model, final_model_path)
    else:
        logging.info("Training model...")
        train_model(model, train_dataloader, optimizer, epochs=model_epochs, batch_size=batch_size,
                    model_save_path=model_save_path, save_model=save_model_flag, save_log=save_log_flag)

    # 使用测试集评估模型性能
    logging.info("Evaluating on test set...")
    evaluate_model(model, test_dataloader)

    # 测试完整句子
    pred_result = predict_sentence(model, tokenizer, test_sentence)

    # 将结果输出为指定格式
    for word, label in pred_result:
        logging.info(f'{word} {label}')