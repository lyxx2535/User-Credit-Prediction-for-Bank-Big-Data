import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == '__main__':

    data = pd.read_csv('credit_to_train.csv')


    X = data[['avg_loan_amt','avg_loan_bal',
              'total_pprd_amotz_intr', 'last_bal',
              'total_tran_amt_huanx', 'total_cac_intc_pr']]  # 选择要计算VIF的字段
    X['intercept'] = 1


    vif = pd.DataFrame()
    vif["Features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


    print(vif)

