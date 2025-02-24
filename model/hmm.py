import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score


# 读取 BIO 格式的 txt 数据
def read_data_from_txt(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        current_sentence = []
        current_labels = []
        for line in lines:
            line = line.strip()
            if line == "":  # 如果遇到空行，跳过
                continue
            word, label = line.split('\t')
            current_sentence.append(word)
            current_labels.append(label)

            if len(current_sentence) > 0 and len(current_labels) > 0:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []
    return sentences, labels


# HMM 模型
class HMM_NER:
    def __init__(self):
        self.states = []  # 状态空间（BIO 标签）
        self.observations = []  # 观测空间（单词）
        self.start_prob = defaultdict(float)  # 初始状态概率
        self.trans_prob = defaultdict(lambda: defaultdict(float))  # 状态转移概率
        self.emit_prob = defaultdict(lambda: defaultdict(float))  # 观测概率（发射概率）

    # 训练 HMM 模型
    def train(self, X_train, y_train):
        state_freq = defaultdict(int)
        observation_freq = defaultdict(int)

        # 计算初始状态概率、状态转移概率和观测概率
        for sentence, labels in zip(X_train, y_train):
            for i, (word, label) in enumerate(zip(sentence, labels)):
                state_freq[label] += 1
                observation_freq[word] += 1

                # 统计初始状态概率
                if i == 0:
                    self.start_prob[label] += 1

                # 统计状态转移概率
                if i > 0:
                    prev_label = labels[i - 1]
                    self.trans_prob[prev_label][label] += 1

                # 统计观测概率
                self.emit_prob[label][word] += 1

        # 归一化初始概率
        total_sentences = len(X_train)
        for state in self.start_prob:
            self.start_prob[state] /= total_sentences

        # 归一化状态转移概率
        for prev_label in self.trans_prob:
            total_trans = sum(self.trans_prob[prev_label].values())
            for label in self.trans_prob[prev_label]:
                self.trans_prob[prev_label][label] /= total_trans

        # 归一化观测概率
        for label in self.emit_prob:
            total_obs = sum(self.emit_prob[label].values())
            for word in self.emit_prob[label]:
                self.emit_prob[label][word] /= total_obs

        self.states = list(state_freq.keys())
        self.observations = list(observation_freq.keys())

    # 维特比算法：找到最优的标签序列
    def viterbi(self, sentence):
        n = len(sentence)
        V = [{}]
        path = {}

        # 初始化维特比算法
        for state in self.states:
            V[0][state] = self.start_prob[state] * self.emit_prob[state].get(sentence[0], 1e-6)
            path[state] = [state]

        # 动态规划步骤
        for t in range(1, n):
            V.append({})
            new_path = {}

            for state in self.states:
                (prob, prev_state) = max((V[t - 1][prev_state] * self.trans_prob[prev_state].get(state, 1e-6) *
                                          self.emit_prob[state].get(sentence[t], 1e-6), prev_state)
                                         for prev_state in self.states)
                V[t][state] = prob
                new_path[state] = path[prev_state] + [state]

            path = new_path

        # 找到最优路径
        (prob, final_state) = max((V[n - 1][state], state) for state in self.states)
        return path[final_state]


# 加载数据集
data_file = '../datasets/bio_data/all_bio_data.txt'  # 修改为你的txt文件路径
X, y = read_data_from_txt(data_file)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 HMM 模型
hmm_model = HMM_NER()
hmm_model.train(X_train, y_train)


# 预测并计算各种评估指标
def evaluate(hmm_model, X_test, y_test):
    y_pred = []
    y_true = []

    for sentence, true_labels in zip(X_test, y_test):
        pred_labels = hmm_model.viterbi(sentence)
        y_pred.extend(pred_labels)  # 将所有预测标签扩展到列表中
        y_true.extend(true_labels)   # 将真实标签扩展到列表中

    # 计算各个指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1, y_true, y_pred


# 计算各项指标并打印
accuracy, precision, recall, f1, y_true, y_pred = evaluate(hmm_model, X_test, y_test)
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

# 打印分类报告
report = classification_report(y_true, y_pred, digits=4, zero_division=0)
print("Classification Report:\n", report)

# 将结果保存到 txt 文件
result_file_path = '../result/hmm_result.txt'
with open(result_file_path, 'w', encoding='utf-8') as result_file:
    result_file.write(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n')
    result_file.write("Classification Report:\n")
    result_file.write(report)