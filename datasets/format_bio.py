# 处理txt文件值不存在但标签存在的无效行
def process_and_save_txt_file(input_file, output_file):
    processed_data = []

    # 处理文件并移除无效行
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去掉首尾空格和换行符
            if line:  # 检查行是否为空
                try:
                    word, label = line.split('\t')  # 以制表符分割
                    if word:  # 检查word是否为空
                        processed_data.append((word, label))
                except ValueError:
                    continue  # 忽略无效行

            else:  # 保留空行
                processed_data.append(("", ""))  # 添加一个占位符，表示空行

    # 将处理后的数据保存到新的txt文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for word, label in processed_data:
            if word or (word == "" and label == ""):  # 保留空行
                file.write(f'{word}\t{label}\n')

    print(f"Processed data saved to {output_file}")


# 使用示例
input_file = './bio_data/inter_bio_data.txt'
output_file = './bio_data/inter_bio_data.txt'
process_and_save_txt_file(input_file, output_file)
