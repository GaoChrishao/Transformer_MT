def process(type, output_path):
    datasets = ["test", "train", "valid"]
    with open(output_path, 'w', encoding='utf-8') as out:
        for dataset in datasets:
            with open("../data/origin_data_set/{}.{}".format(dataset, type), 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                while line:

                    line = line.replace("@@ ", "")
                    out.write(line + "\n")
                    line = f.readline().strip()





if __name__ == '__main__':
    # 将课程提供的数据集进行初步处理（去除@@符号，并合并train,valid,test）
    process("de", "../data/de.txt")
    process("en", "../data/en.txt")

    print("处理完成！")
