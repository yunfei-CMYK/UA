import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from TorchCRF import CRF
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

# 禁用 cudnn 对于 RNN 的优化，避免在某些情况下反向传播出错
cudnn.enabled = False


# 配置类
class Config:
    def __init__(self):
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        self.model_save_path = './saved_model/new_bilstm_crf.pt'
        self.log_file = '../result/new_bilstm_crf.txt'


config = Config()


# 定义 BiLSTM + CRF 模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(config.hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size)

    def forward(self, sentences, tags, mask):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)
        # 对 CRF 的损失取绝对值，确保不为负
        loss = torch.abs(-self.crf(emissions, tags, mask=mask.bool()))
        return loss.mean()

    def predict(self, sentences, mask):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)
        return self.crf.viterbi_decode(emissions, mask=mask.bool())


# 自定义数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]


# 数据加载函数
def read_data(file_path):
    sentences = []
    tags = []
    sentence = []
    tag_sequence = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                word, label = line.strip().split()
                sentence.append(word)
                tag_sequence.append(label)
            else:
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_sequence)
                    sentence = []
                    tag_sequence = []
        if sentence:
            sentences.append(sentence)
            tags.append(tag_sequence)

    return sentences, tags


# 填充函数
def pad_sequences(sequences, max_len=None, padding_value=0):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.full((len(sequences), max_len), padding_value)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_sequences[i, :length] = seq
    return padded_sequences


# 自定义 collate_fn
def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences = pad_sequences(sentences)
    tags = pad_sequences(tags)
    mask = (sentences != 0)
    return torch.tensor(sentences, dtype=torch.long), torch.tensor(tags, dtype=torch.long), torch.tensor(mask,
                                                                                                         dtype=torch.bool)


# 准备数据
train_sentences, train_tags = read_data('../datasets/bio_data/all_bio_data.txt')
word_vocab = {word: idx for idx, word in enumerate(set(word for sent in train_sentences for word in sent))}
tag_vocab = {tag: idx for idx, tag in enumerate(set(tag for tags in train_tags for tag in tags))}
train_sentences_idx = [[word_vocab[word] for word in sentence] for sentence in train_sentences]
train_tags_idx = [[tag_vocab[tag] for tag in tags] for tags in train_tags]

# 拆分数据集，按8:1:1的比例
train_sentences_idx, temp_sentences_idx, train_tags_idx, temp_tags_idx = train_test_split(
    train_sentences_idx, train_tags_idx, test_size=0.2, random_state=42
)

# 再次拆分临时集为验证集和测试集
val_sentences_idx, test_sentences_idx, val_tags_idx, test_tags_idx = train_test_split(
    temp_sentences_idx, temp_tags_idx, test_size=0.5, random_state=42
)

# 打印每个数据集的大小
print(f'Train dataset size: {len(train_sentences_idx)}')
print(f'Validation dataset size: {len(val_sentences_idx)}')
print(f'Test dataset size: {len(test_sentences_idx)}')


# 创建 DataLoader
train_loader = DataLoader(NERDataset(train_sentences_idx, train_tags_idx), config.batch_size, shuffle=True,
                          collate_fn=collate_fn)

val_loader = DataLoader(NERDataset(val_sentences_idx, val_tags_idx), config.batch_size, shuffle=False,
                        collate_fn=collate_fn)

test_loader = DataLoader(NERDataset(test_sentences_idx, test_tags_idx), config.batch_size, shuffle=False,
                         collate_fn=collate_fn)


# 训练模型并保存日志
def train(model, train_loader, val_loader, optimizer, epochs, model_save_path, log_file):
    print("Training model...")
    model.train()  # 确保模型处于训练模式
    with open(log_file, 'w') as f_log:
        f_log.write("Epoch, Loss, Accuracy, Precision, Recall, F1\n")
        for epoch in range(epochs):
            total_loss = 0.0
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True) as pbar:
                for sentences, tags, mask in pbar:
                    sentences, tags, mask = sentences.cuda(), tags.cuda(), mask.cuda()  # 数据迁移到GPU
                    optimizer.zero_grad()
                    loss = model(sentences, tags, mask)
                    loss.backward()  # 计算梯度
                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪梯度
                    optimizer.step()  # 更新模型参数
                    total_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

            # 评估验证集
            all_tags, all_preds = evaluate(model, val_loader)

            # 计算指标
            accuracy = accuracy_score(all_tags, all_preds)
            precision = precision_score(all_tags, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_tags, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_tags, all_preds, average='weighted', zero_division=0)

            # 记录日志
            f_log.write(f'{epoch + 1}, {total_loss:.6f}, {accuracy:.4f}, {precision:.4f}, {recall:.4f}, {f1:.4f}\n')
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}, '
                  f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                  f'Recall: {recall:.4f}, F1: {f1:.4f}')
            pbar.close()

        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')



# 评估模型
def evaluate(model, val_loader):
    model.eval()
    all_preds, all_tags = [], []
    with torch.no_grad():
        for sentences, tags, mask in val_loader:
            sentences, tags, mask = sentences.cuda(), tags.cuda(), mask.cuda()  # 数据迁移到GPU
            preds = model.predict(sentences, mask)
            for pred, tag, m in zip(preds, tags, mask):
                valid_pred = [p for p, mask_val in zip(pred, m) if mask_val]
                valid_tag = [t.item() for t, mask_val in zip(tag, m) if mask_val]
                if len(valid_pred) != len(valid_tag):
                    continue
                all_preds.extend(valid_pred)
                all_tags.extend(valid_tag)
    return all_tags, all_preds


# 测试集评估模型
def evaluate_test(model, test_loader):
    model.eval()
    all_preds, all_tags = [], []
    with torch.no_grad():
        for sentences, tags, mask in test_loader:
            sentences, tags, mask = sentences.cuda(), tags.cuda(), mask.cuda()  # 数据迁移到GPU
            preds = model.predict(sentences, mask)
            for pred, tag, m in zip(preds, tags, mask):
                valid_pred = [p for p, mask_val in zip(pred, m) if mask_val]
                valid_tag = [t.item() for t, mask_val in zip(tag, m) if mask_val]
                if len(valid_pred) != len(valid_tag):
                    continue
                all_preds.extend(valid_pred)
                all_tags.extend(valid_tag)
    return all_tags, all_preds


# 预测句子
def predict_sentence(model, sentence, word_vocab, tag_vocab):
    model.eval()
    words = list(sentence)
    sentence_idx = [word_vocab.get(word, 0) for word in words]  # 将句子转换为索引
    sentence_tensor = torch.tensor([sentence_idx], dtype=torch.long).cuda()  # 转换为张量并迁移到GPU
    mask = torch.tensor([[1] * len(sentence_idx)], dtype=torch.bool).cuda()  # 生成掩码并迁移到GPU
    with torch.no_grad():
        pred_tags = model.predict(sentence_tensor, mask)[0]  # 获取预测标签
    id2tag = {v: k for k, v in tag_vocab.items()}
    pred_labels = [id2tag[p] for p in pred_tags]
    for word, label in zip(words, pred_labels):
        print(f'{word} {label}')

# 主函数
def main():
    vocab_size = len(word_vocab)
    tagset_size = len(tag_vocab)

    model = BiLSTM_CRF(vocab_size, tagset_size).cuda()  # 模型迁移到GPU
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    if os.path.exists(config.model_save_path):
        model.load_state_dict(torch.load(config.model_save_path))
        print("Loading existing model...")
    else:
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
        train(model, train_loader, val_loader, optimizer, epochs=config.epochs,
              model_save_path=config.model_save_path, log_file=config.log_file)

    # 在测试集上进行评估
    test_tags, test_preds = evaluate_test(model, test_loader)

    accuracy = accuracy_score(test_tags, test_preds)
    precision = precision_score(test_tags, test_preds, average='weighted', zero_division=0)
    recall = recall_score(test_tags, test_preds, average='weighted', zero_division=0)
    f1 = f1_score(test_tags, test_preds, average='weighted', zero_division=0)

    print(f'Test Set Evaluation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

    test_sentence = "民用无人机大范围使用,其中包含了大量的传感器，比如姿态传感器。"
    predict_sentence(model, test_sentence, word_vocab, tag_vocab)

if __name__ == "__main__":
    main()
