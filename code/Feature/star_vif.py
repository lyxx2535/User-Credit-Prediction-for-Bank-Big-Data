import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == '__main__':

    data = pd.read_csv('star_to_train.csv')


    X = data[['all_bal_summary','avg_month_summary',
              'avg_year_summary',
              'fixed_deposit_summary']]  # 选择要计算VIF的字段
    X['intercept'] = 1


    vif = pd.DataFrame()
    vif["Features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


    print(vif)

