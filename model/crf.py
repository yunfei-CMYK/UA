import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from TorchCRF import CRF
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm  # 引入 tqdm


# 自定义数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


# 加载数据集
def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        current_labels = []
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                word, label = line.split()
                current_sentence.append(word)
                current_labels.append(label)

        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)

    return sentences, labels


# 将句子和标签转换为索引
def convert_to_index(sentences, labels, word_to_index, label_to_index):
    indexed_sentences = []
    indexed_labels = []

    for sentence, label in zip(sentences, labels):
        indexed_sentences.append([word_to_index.get(word, 0) for word in sentence])  # OOV 词处理
        indexed_labels.append([label_to_index[lbl] for lbl in label])

    # 填充为相同长度
    indexed_sentences = pad_sequence([torch.tensor(seq) for seq in indexed_sentences], batch_first=True,
                                     padding_value=0)
    indexed_labels = pad_sequence([torch.tensor(seq) for seq in indexed_labels], batch_first=True,
                                  padding_value=-1)  # 标签填充使用 -1

    return indexed_sentences, indexed_labels


# 定义 CRF 模型
class CRFModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(CRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, num_classes)
        self.crf = CRF(num_classes)

    def forward(self, x, tags=None):
        embedded = self.embedding(x)
        emissions = self.fc(embedded)

        # 创建 mask，mask 的形状是 [batch_size, max_length]
        mask = (x != 0)  # 假设填充索引为 0

        if tags is not None:
            return -self.crf(emissions, tags, mask=mask)
        else:
            return self.crf.viterbi_decode(emissions, mask=mask)


# 训练函数
def train(model, data_loader, optimizer, num_epochs, device=None):
    model.train()
    results_file_path = '../result/new_crf_result.txt'
    with open(results_file_path, 'w') as results_file:
        results_file.write("Epoch, Loss, Accuracy, Precision, Recall, F1 Score\n")
    for epoch in range(num_epochs):
        total_loss = 0
        all_predictions = []
        all_true_labels = []

        with tqdm(data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for x, tags in tepoch:
                # 将输入数据和标签移动到设备
                x = x.to(device)
                tags = tags.to(device)

                optimizer.zero_grad()
                loss = model(x, tags)

                loss = loss.mean()  # 求平均，确保 loss 是标量

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # 收集预测和真实标签
                predictions = model(x)
                for pred, true in zip(predictions, tags):
                    true_filtered = true[true != -1]
                    pred = torch.tensor(pred, device=device)
                    all_predictions.extend(pred.tolist())
                    all_true_labels.extend(true_filtered.tolist())

                tepoch.set_postfix(loss=total_loss / len(data_loader))

        # 计算损失
        avg_loss = total_loss / len(data_loader)
        # 计算准确率、精确率、召回率和 F1 值
        accuracy = (torch.tensor(all_predictions) == torch.tensor(all_true_labels)).float().mean().item()
        precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='weighted')
        f1 = f1_score(all_true_labels, all_predictions, average='weighted')

        print(
            f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

        # 将结果写入文件
        with open(results_file_path, 'a') as results_file:
            results_file.write(
                f'{epoch + 1},{total_loss / len(data_loader):.4f},{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n')

    return model


# 测试函数
def evaluate(model, data_loader, device=None):
    model.eval()
    total_correct = 0
    total_tokens = 0
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for x, tags in data_loader:
            # 将输入数据移动到设备
            x = x.to(device)
            tags = tags.to(device)

            # 获取预测值，假设 model(x) 返回解码后的预测标签
            predictions = model(x)

            # 遍历批次中的每个预测序列和对应的标签序列
            for pred, true in zip(predictions, tags):
                # 将 `true` 中的填充部分过滤掉（假设填充标签为 -1）
                true_filtered = true[true != -1]

                # 计算正确的预测数量
                pred = torch.tensor(pred, device=device)  # 将预测结果转换为张量，保持数据类型一致
                total_correct += (pred == true_filtered).sum().item()
                total_tokens += len(true_filtered)
                all_predictions.extend(pred.tolist())
                all_true_labels.extend(true_filtered.tolist())

    accuracy = total_correct / total_tokens
    precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1


def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences = pad_sequence([torch.as_tensor(seq) for seq in sentences], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.as_tensor(seq) for seq in labels], batch_first=True, padding_value=-1)
    return sentences, labels


# 主函数
def main():
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = '../datasets/bio_data/all_bio_data.txt'  # 替换为您的 BIO 数据集路径
    sentences, labels = load_data(file_path)

    # 创建词汇表和标签映射
    word_set = {word for sentence in sentences for word in sentence}
    label_set = {label for label_list in labels for label in label_list}

    word_to_index = {word: idx + 1 for idx, word in enumerate(word_set)}  # 从 1 开始
    label_to_index = {lbl: idx for idx, lbl in enumerate(label_set)}

    # 将句子和标签转换为索引
    indexed_sentences, indexed_labels = convert_to_index(sentences, labels, word_to_index, label_to_index)

    # 构建数据集
    dataset = NERDataset(indexed_sentences, indexed_labels)

    # 数据集划分：8:1:1
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f'Train dataset size: {train_size}')
    print(f'Validation dataset size: {val_size}')
    print(f'Test dataset size: {test_size}')
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # 模型参数
    vocab_size = len(word_to_index) + 1
    embed_size = 128
    num_classes = len(label_to_index)

    # 初始化模型
    model = CRFModel(vocab_size, embed_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print(f'Training on {device}')
    trained_model = train(model, train_loader, optimizer, num_epochs=100, device=device)

    # 保存模型
    torch.save(trained_model.state_dict(), './saved_model/crf.pt')
    print("Model saved.")

    # 在测试集上评估
    print("Testing the model:")
    accuracy, precision, recall, f1 = evaluate(trained_model, test_loader, device=device)
    print(
        f'Test Set Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')


if __name__ == "__main__":
    main()
