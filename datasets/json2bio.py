import json


# 从JSON文件中读取数据
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 将JSON转换为BIO格式
def json_to_bio(data):
    bio_data = []
    for entry in data:
        text = entry.get('text', "")
        labels = entry.get('label', [])

        # 初始化BIO列表，所有字符的默认标签为'O'
        bio_tags = ['O'] * len(text)

        # 检查是否有label字段，并且非空
        if not labels:
            continue  # 如果没有label，跳过该条目

        # 遍历标注的标签
        for label in labels:
            start = label['start']
            end = label['end']
            label_type = label['labels'][0]  # 使用第一个标签作为类型

            # 标记B和I
            bio_tags[start] = f'B-{label_type}'
            for i in range(start + 1, end):
                bio_tags[i] = f'I-{label_type}'

        # 生成BIO格式的结果
        for i, char in enumerate(text):
            bio_data.append(f"{char}\t{bio_tags[i]}")
        bio_data.append("")  # 添加空行，分隔每一段文本

    return "\n".join(bio_data)


# 将转换后的BIO格式数据保存为txt文件
def save_to_txt(file_path, bio_format):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(bio_format)


# 主函数
def convert_json_to_bio_file(input_json_file, output_txt_file):
    # 读取JSON文件
    data = read_json_file(input_json_file)

    # 将JSON转换为BIO格式
    bio_format = json_to_bio(data)

    # 保存BIO格式到txt文件
    save_to_txt(output_txt_file, bio_format)
    print(f"BIO格式数据已保存到 {output_txt_file}")


# 示例用法
input_json_file = './json_data/inter_jsondata.json'  # 输入的JSON文件
output_txt_file = './bio_data/inter_bio_data.txt'  # 输出的txt文件

convert_json_to_bio_file(input_json_file, output_txt_file)
