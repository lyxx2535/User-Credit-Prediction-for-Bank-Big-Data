import pandas as pd
from scipy.stats import chi2_contingency

if __name__ == '__main__':
    # 读取数据集
    df = pd.read_csv('credit_to_train.csv')

    # 计算phi系数
    crosstab = pd.crosstab(df['is_withdrw'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['is_transfer'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_employee = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
                   ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_employee = max(0, min((phi_employee**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['is_deposit'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_shareholder = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
                      ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_shareholder = max(0, min((phi_shareholder**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['is_purchse'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_black = ((crosstab.iloc[0, 0]*crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1]*crosstab.iloc[1, 0])) / \
                ((crosstab.iloc[0, 0]+crosstab.iloc[0, 1])*(crosstab.iloc[1, 0]+crosstab.iloc[1, 1])) ** 0.5
    phi_black = max(0, min((phi_black**2/n)**0.5, 1))

    crosstab = pd.crosstab(df['is_mob_bank'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_black = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
                ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_black = max(0, min((phi_black ** 2 / n) ** 0.5, 1))

    crosstab = pd.crosstab(df['is_etc'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex ** 2 / n) ** 0.5, 1))

    crosstab = pd.crosstab(df['is_employee'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex ** 2 / n) ** 0.5, 1))

    crosstab = pd.crosstab(df['is_shareholder'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex ** 2 / n) ** 0.5, 1))

    crosstab = pd.crosstab(df['is_black'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex ** 2 / n) ** 0.5, 1))

    crosstab = pd.crosstab(df['is_contact'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex ** 2 / n) ** 0.5, 1))

    crosstab = pd.crosstab(df['is_mgr_dep'], df['credit_level'])
    _, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    phi_sex = ((crosstab.iloc[0, 0] * crosstab.iloc[1, 1]) - (crosstab.iloc[0, 1] * crosstab.iloc[1, 0])) / \
              ((crosstab.iloc[0, 0] + crosstab.iloc[0, 1]) * (crosstab.iloc[1, 0] + crosstab.iloc[1, 1])) ** 0.5
    phi_sex = max(0, min((phi_sex ** 2 / n) ** 0.5, 1))

    print('phi系数：')
    print(f'is_withdrw与credit_level的phi系数为{phi_sex:.2f}')
    print(f'is_transfer与credit_level的phi系数为{phi_employee:.2f}')
    print(f'is_deposit与credit_level的phi系数为{phi_shareholder:.2f}')
    print(f'is_purchse与credit_level的phi系数为{phi_black:.2f}')
    print(f'is_mob_bank与credit_level的phi系数为{phi_black:.2f}')
    print(f'is_etc与credit_level的phi系数为{phi_black:.2f}')
    print(f'is_employee与credit_level的phi系数为{phi_black:.2f}')
    print(f'is_shareholder与credit_level的phi系数为{phi_black:.2f}')
    print(f'is_black与credit_level的phi系数为{phi_black:.2f}')
    print(f'is_contact与credit_level的phi系数为{phi_black:.2f}')
    print(f'is_mgr_dep与credit_level的phi系数为{phi_black:.2f}')