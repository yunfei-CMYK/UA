def merge_bio_files(file1, file2, output_file):
    """
    将两个BIO格式的txt文件合并，并保存到指定的输出文件中。

    参数:
    file1: 第一个文件的路径
    file2: 第二个文件的路径
    output_file: 输出文件路径
    """
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        content1 = f1.readlines()  # 读取第一个文件的内容
        content2 = f2.readlines()  # 读取第二个文件的内容

    # 合并文件内容，将第二个文件的内容追加到第一个文件的内容后面
    combined_content = content1 + content2

    # 将合并后的内容写入输出文件
    with open(output_file, 'w', encoding='utf-8') as output:
        output.writelines(combined_content)

    print(f"两个文件已成功合并，并保存为: {output_file}")


# 示例用法
file1 = './bio_data/bio_data.txt'  # 替换为第一个BIO格式文件的路径
file2 = './bio_data/inter_bio_data.txt'  # 替换为第二个BIO格式文件的路径
output_file = './bio_data/bio_data.txt'  # 替换为合并后的文件路径

merge_bio_files(file1, file2, output_file)
