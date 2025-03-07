import pandas as pd


def merge_csv_files(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv(output_file, index=False)


def delete_csv_column(input_file, output_file, column_index):
    df = pd.read_csv(input_file)
    df.drop(df.columns[column_index], axis=1, inplace=True)
    df.to_csv(output_file, index=False)


def move_column_to_last_row(input_file, output_file, column_index):
    df = pd.read_csv(input_file)
    column_values = df.iloc[:, column_index]  # 获取要移动的列的值
    last_row = df.iloc[-1, :]  # 获取最后一行

    last_row = last_row.append(column_values, ignore_index=True)  # 在最后一行中添加列值
    df = df.iloc[:-1, :]  # 删除最后一行
    df = df.append(last_row, ignore_index=True)  # 将新的最后一行添加到DataFrame

    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    # 指定要合并的两个CSV文件以及输出文件的路径
    file1_path = '../data/star_to_train.csv'
    file2_path = '../data/star_to_fill.csv'
    output_file_path = 'star.csv'

    # 调用函数进行合并
    merge_csv_files(file1_path, file2_path, output_file_path)

    # 指定输入和输出文件路径以及要删除的列索引（从0开始）
    # input_file_path = '../data/star_to_fill.csv'
    # output_file_path = '../data/star_to_fill.csv'
    # column_index = 1
    #
    # move_column_to_last_row(input_file_path, output_file_path, column_index)
