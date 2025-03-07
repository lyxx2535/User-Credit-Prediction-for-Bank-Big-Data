import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 读取 CSV 文件
    credit = pd.read_csv('credit.csv')
    star = pd.read_csv('star.csv')

    # 对于每个数据集的每个数值列，计算并打印统计信息，生成新的CSV文件，并创建直方图
    for df, name in zip([credit, star], ['Credit', 'Star']):
        print(f"\n{name} Dataset:")
        print("\nFirst 5 rows:")
        print(df.head())

        # 选择数值列
        num_cols = df.select_dtypes(include=[np.number]).columns

        # 计算描述性统计信息，并输出为新的CSV文件
        desc_df = df[num_cols].describe().transpose()
        desc_df.to_csv(f'{name}_describe.csv')
        print(f"\nDescriptive statistics for {name} Dataset has been written to {name}_describe.csv")

        print("\nFrequency Count and Histograms:")
        for col in num_cols:
            # 计算并打印频数
            print(f"\nColumn: {col}")
            print(df[col].value_counts())

            # 创建直方图并在直方图上标注每个频数的具体值
            plt.figure(figsize=(10, 6))
            plot = sns.histplot(df[col], kde=False, bins=30)
            for p in plot.patches:
                plot.annotate(f'{p.get_height():.0f}',
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center',
                              xytext=(0, 10),
                              textcoords='offset points')
            plt.title(f"Histogram for {col} in {name} Dataset")
            plt.savefig(f'Histogram for {col} in {name} Dataset.png')
            plt.show()

