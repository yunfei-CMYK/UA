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

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.data.view(-1))
            F_loss = at * F_loss.view(-1)
        return F_loss.mean()

# 定义自适应特征融合模块（AFFM）
class AdaptiveFeatureFusionModule(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveFeatureFusionModule, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, bert_features, bilstm_features):
        weights = self.sigmoid(self.weight)
        fused_features = weights * bert_features + (1 - weights) * bilstm_features
        return fused_features

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
            if line == "":
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
                    continue
        if current_sentence:
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
    def __init__(self, bert_model_name, num_labels, hidden_dim=256):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        # 添加线性层，将 BiLSTM 输出维度转换为 BERT 输出维度
        self.proj = nn.Linear(hidden_dim * 2, self.bert.config.hidden_size)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)
        self.focal_loss = FocalLoss(gamma=2, alpha=None)  # 初始化焦点损失函数
        self.affm = AdaptiveFeatureFusionModule(self.bert.config.hidden_size)

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取BERT输出
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # BiLSTM层
        lstm_output, _ = self.lstm(sequence_output)

        # 投影 BiLSTM 输出到与 BERT 相同的维度
        lstm_output = self.proj(lstm_output)

        # 自适应特征融合
        fused_features = self.affm(sequence_output, lstm_output)

        # 线性层
        emissions = self.fc(fused_features)

        if labels is not None:
            # 使用焦点损失函数计算损失
            emissions_flat = emissions.view(-1, emissions.size(-1))
            labels_flat = labels.view(-1)
            loss = self.focal_loss(emissions_flat, labels_flat)
            return loss
        else:
            # CRF层 (推理模式)
            prediction = self.crf.viterbi_decode(emissions, mask=attention_mask.bool())  # 使用 .bool()
            return prediction

# 训练模型，只在最后一次epoch后保存模型
def train_model(model, dataloader, optimizer, epochs, batch_size, model_save_path, save_log, save_model):
    model.train()

    if save_log:
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

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        if save_log:
            logging.info(f'{epoch + 1}, {avg_loss}, {accuracy}, {precision}, {recall}, {f1}')

    if save_model:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        final_model_path = os.path.join(model_save_path, 'bert_bilstm_crf_focal_loss_AFFM.pt')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Model saved after final epoch at {final_model_path}")

# 加载保存的模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 测试完整句子的识别
def predict_sentence(model, tokenizer, sentence, max_len):
    model.eval()
    sentence_tokens = list(sentence)

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

    pred_labels = preds[0]  # 只取第一个样本（单句）
    idx2tag = {v: k for k, v in dataset.label_map.items()}
    pred_tags = [idx2tag.get(p, 'O') for p in pred_labels]
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
                valid_length = mask.sum().item()
                all_predictions.extend(pred[:valid_length])
                all_labels.extend(label[:valid_length].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    logging.info(f'Test Set Evaluation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

if __name__ == "__main__":
    # 参数配置
    data_file = '../datasets/bio_data/all_bio_data.txt'
    bert_model_name = 'bert-base-chinese'
    model_epochs = 1
    batch_size = 32
    hidden_dim = 256
    max_len = 256
    model_save_path = './saved_model'
    test_sentence = '民用无人机大范围使用，其中包含了大量的传感器，比如姿态传感器。'

    # 添加是否保存日志和模型的开关
    save_log_flag = True  # 设置为 False，日志不应保存到文件
    save_model_flag = False

    # 动态配置日志记录器
    log_dir = '../log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    script_name = os.path.basename(__file__)
    log_file = os.path.join(log_dir, f"{current_date}_{script_name.replace('.py', '.log')}")

    handlers = [logging.StreamHandler()]  # 默认只输出到控制台
    if save_log_flag:
        handlers.append(logging.FileHandler(log_file))  # 如果需要保存日志，则添加文件处理器

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

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
    dataset = NERDataset(sentences, labels, tokenizer, max_len)

    # 使用 random_split 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 记录每个数据集的大小
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
    final_model_path = os.path.join(model_save_path, 'bert_bilstm_crf_focal_loss_AFFM.pt')
    if os.path.exists(final_model_path):
        logging.info("Loading existing model...")
        model = load_model(model, final_model_path)
    else:
        logging.info("Training model...")
        train_model(model, train_dataloader, optimizer, epochs=model_epochs, batch_size=batch_size,
                    model_save_path=model_save_path, save_log=save_log_flag, save_model=save_model_flag)

    # 使用测试集评估模型性能
    logging.info("Evaluating on test set...")
    evaluate_model(model, test_dataloader)

    # 测试完整句子
    pred_result = predict_sentence(model, tokenizer, test_sentence, max_len)

    # 将结果输出为指定格式
    for word, label in pred_result:
        logging.info(f'{word} {label}')