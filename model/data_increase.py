import torch
from transformers import BertTokenizerFast
import os

# 假设 BERT_BiLSTM_CRF 类和其他必要的函数在另一个模块中定义
# 如果它们在同一个脚本中，可以省略这个导入语句。
from bert_bilstm_crf import BERT_BiLSTM_CRF, load_model, predict_sentence  # 将 'your_module' 替换为实际模块名称

# 加载模型
model_save_path = './saved_model/bert_bilstm_crf.pt'
model = BERT_BiLSTM_CRF('bert-base-chinese', num_labels=33)  # 根据需要更新 num_labels
model.load_state_dict(torch.load(model_save_path))
model.eval()
model.to('cuda')

# 加载分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

# 从文件中读取句子的函数
def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# 将预测结果写入文件的函数
def write_predictions_to_file(predictions, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            for word, label in pred:
                f.write(f'{word} {label}\n')
            f.write('\n')  # 在每个预测结果之间添加空行

# 主函数，用于对文件中的句子进行预测
def main(input_file_path, output_file_path):
    sentences = read_sentences_from_file(input_file_path)
    all_predictions = []

    for sentence in sentences:
        pred_result = predict_sentence(model, tokenizer, sentence)
        all_predictions.append(pred_result)

    write_predictions_to_file(all_predictions, output_file_path)
    print(f"预测结果已保存到 {output_file_path}")

# 定义输入和输出文件路径
input_file = '../datasets/origin_data/common_sense_data.txt'  # 替换为你的输入文件路径
output_file = '../datasets/bio_data/common_bio_data.txt'  # 替换为你想要的输出文件路径

# 执行主函数
main(input_file, output_file)
