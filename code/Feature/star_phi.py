import pandas as pd
from scipy.stats import chi2_contingency

if __name__ == '__main__':
    # 读取数据集
    df = pd.read_csv('star_to_train.csv')

    # 计算phi系数
    crosstab = pd.crosstab(df['sex'], df['star_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['is_employee'], df['star_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_employee = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
                   ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_employee = max(0, min((phi_employee**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['is_shareholder'], df['star_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_shareholder = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
                      ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_shareholder = max(0, min((phi_shareholder**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['is_black'], df['star_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_black = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
                ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_black = max(0, min((phi_black**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['djk_info_max_purchase_function'], df['star_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex ** 2 / n) ** 0.5, 1))

    print('phi系数：')
    print(f'sex与star_level的phi系数为{phi_sex:.2f}')
    print(f'is_employee与star_level的phi系数为{phi_employee:.2f}')
    print(f'is_shareholder与star_level的phi系数为{phi_shareholder:.2f}')
    print(f'is_black与star_level的phi系数为{phi_black:.2f}')
    print(f'djk_info_max_purchase_function与star_level的phi系数为{phi_black:.2f}')