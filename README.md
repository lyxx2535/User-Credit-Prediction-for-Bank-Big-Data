## 0、项目结构

img——数据盘点的各项图片

code——

DataPreprocessing——预处理相关

Feature——特征相关

train.py——模型训练选择与应用

res——得到结果csv

data——重要中间数据csv

## 1、数据获取

### a. 数据获取与字段选择

以下是我分析的表格，以评估star_level和credit_level：

#### 与star_level有关的表格：

1. pri_cust_asset_info '存款汇总信息'
   1. `all_bal`（总余额）: 这个字段可以代表客户的总存款量，一般来说，存款越多的客户可能对银行的服务需求和要求更高，相应的星级也可能更高。
   2. `avg_mth, avg_qur, avg_year`（月/季度/年日均）: 这些字段反映了客户的存款在一定周期内的平均水平，可以帮助我了解客户的存款稳定性和变动情况，这可能影响他们的星级。
   3. `sa_bal, td_bal, fin_bal`（活期余额，定期余额，理财余额）: 这些字段显示了客户的资金配置，对我判断其风险承受能力和服务需求有所帮助，可能影响星级。

![image-20230515175020651](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537567.png)

2. pri_cust_base_info '基本信息'

   1. `avg_acct_bal_account` -- '存款账号信息.账户余额均值'，客户所有账户的平均余额。这一数据体现了客户的资金状况和储蓄习惯，有助于我了解其财务稳定性，从而进行合适的星级评定。
   2. `avg_bal_account` -- '存款账号信息.原币余额均值'，客户所有账户的平均原币余额。此项信息提供了客户账户原币的平均值，这有助于我理解客户的国际财务交易和资金流动情况，有利于提供更个性化的服务和产品。
   3. `avg_avg_mth_account` -- '存款账号信息.月日均均值'，客户所有账户的平均月日均存款余额。通过观察客户的月度平均存款，我可以把握其月度收支波动，并据此评估其财务管理能力，作为星级评定的一个参考因素。
   4. `avg_avg_qur_account` -- '存款账号信息.季度日均均值'，客户所有账户的平均季度日均存款余额。这一信息揭示了客户在季度级别上的储蓄习惯和财务状况，帮助我预测他们的长期存款趋势，以优化我的营销策略。
   5. `avg_avg_year_account` -- '存款账号信息.年日均均值'，客户所有账户的平均年日均存款余额。通过研究年度数据，我可以更好地理解客户的长期财务状况，以及其储蓄和投资策略，从而更准确地评定其星级。
   6. `sex` -- 客户性别，1表示男性，0表示女性，可能会根据银行规则不同有一些星级评定的区别。
   7. `is_employee` -- 客户是否是员工，1表示是，0表示不是。员工可能有不同于普通客户的需求和行为，理解这一差异可以帮助我为员工提供更优质的服务，并提高他们的忠诚度。
   8. `is_shareholder` -- 客户是否是股东，1表示是，0表示不是。股东作为公司的所有者，他们的意见和满意度至关重要，为股东提供优质服务，不仅可以保持他们的忠诚度，还有助于提升公司形象。
   9. `is_black` -- 客户是否在黑名单中，1表示是，0表示不是。了解客户的信用情况对于风险管理至关重要，此信息可以帮助我避免潜在的风险，保护公司利益，同时也为客户提供合适的服务和建议。

   ![image-20230515210746617](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537577.png)

3. pri_cust_asset_acct_info '存款账号信息' 

   1. `acct_bal`和`bal`：账户的余额可能是影响星级的重要因素。
   2. `avg_mth`，`avg_qur`，`avg_year`：这些字段和之前分析的一样，可以反映出用户的存款稳定性和变动情况，这可能影响星级。

   ![image-20230515175135556](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537570.png)

   

4. djk_info '贷记卡开户明细'

   1. `is_withdrw`：是否开通取现功能可能反映了客户的贷记卡使用习惯和需求。
   2. `is_transfer`：是否开通转账功能也可能反映了客户的贷记卡使用习惯和需求。
   3. `is_deposit`：是否开通存款功能可能反映了客户的贷记卡使用习惯和需求，可能影响他们的星级。
   4. `is_purchse`：是否开通消费功能可能反映了客户的贷记卡使用习惯和需求，可能影响他们的星级。
   5. `cred_limit`：信用额度可能反映了业务审批员对客户的信任程度，可能影响他们的星级。
   6. `dlay_amt`：逾期金额可能反映了客户的信用状况，可能影响他们的星级。
   7. `bal`：余额可能反映了客户的财务状况，可能影响他们的星级。


![image-20230516002827317](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537555.png)

这些表格可以提供与客户存款和日常交易有关的信息，有助于评估客户的star_level。

最后，我的表结构为

```
star '客户存款与贷记卡信息'(
uid -- 证件号码，唯一标识客户
star_level -- 客户星级，主要表示客户的价值或贡献大小
cust_no -- 核心客户号，客户在银行系统中的唯一标识号
cust_name -- 客户名称
all_bal_summary -- '存款汇总信息.总余额'，客户在银行所有账户的存款总余额
avg_month_summary -- '存款汇总信息.月日均'，客户在银行所有账户的月日均存款余额
avg_quarter_summary -- '存款汇总信息.季度日均'，客户在银行所有账户的季度日均存款余额
avg_year_summary -- '存款汇总信息.年日均'，客户在银行所有账户的年日均存款余额
current_deposit_summary -- '存款汇总信息.活期余额'，客户的所有活期存款余额
fixed_deposit_summary -- '存款汇总信息.定期余额'，客户的所有定期存款余额
financial_management_balance_summary -- '存款汇总信息.理财余额'，客户所有理财产品的余额
avg_acct_bal_account -- '存款账号信息.账户余额均值'，客户所有账户的平均余额
avg_bal_account -- '存款账号信息.原币余额均值'，客户所有账户的平均原币余额
avg_avg_mth_account -- '存款账号信息.月日均均值'，客户所有账户的平均月日均存款余额
avg_avg_qur_account -- '存款账号信息.季度日均均值'，客户所有账户的平均季度日均存款余额
avg_avg_year_account -- '存款账号信息.年日均均值'，客户所有账户的平均年日均存款余额
sex -- 客户性别，1表示男性，0表示女性
is_employee -- 客户是否是员工，1表示是，0表示不是
is_shareholder -- 客户是否是股东，1表示是，0表示不是
is_black -- 客户是否在黑名单中，1表示是，0表示不是
djk_info_max_withdrawal_function -- '贷记卡开户明细.是否开通取现功能'，1表示开通，0表示未开通
djk_info_max_transfer_function -- '贷记卡开户明细.是否开通转账功能'，1表示开通
djk_info_max_transfer_function -- '贷记卡开户明细.是否开通转账功能'，1表示开通，0表示未开通
djk_info_max_deposit_function -- '贷记卡开户明细.是否开通存款功能'，1表示开通，0表示未开通
djk_info_max_purchase_function -- '贷记卡开户明细.是否开通消费功能'，1表示开通，0表示未开通
djk_info_max_credit_limit -- '贷记卡开户明细.信用额度'，在所有贷记卡中的最大信用额度
djk_info_max_delay_amt -- '贷记卡开户明细.逾期金额'，在所有贷记卡中的最大逾期金额
djk_info_max_bal -- '贷记卡开户明细.余额'，在所有贷记卡中的最大余额)
```



#### 与credit_level有关的表格：

1. pri_cust_liab_info '贷款账户汇总'

   - `all_bal`：这是客户所有贷款的总余额，它可以反映客户的贷款负担。
   - `bad_bal`：这是不良贷款余额，它可以反映客户的信用状况。
   - `due_intr`：这是欠息总额，可以反映客户的贷款偿还情况。
   - `norm_bal`：这是正常贷款余额，可以反映客户的贷款结构。
   - `delay_bal`：这是逾期贷款总额，它可以反映客户的贷款偿还情况。

   ![image-20230516152445909](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537552.png)

2. pri_cust_liab_acct_info '贷款账号信息'

   1. `loan_amt`：贷款金额，一个重要的信贷特征。
   2. `loan_bal`：贷款余额，也是一个重要的信贷特征。
   3. `is_mortgage`：是否按揭，这个字段可以表示客户的负债类型。
   4. `is_online`：是否线上贷款，这个字段可以表示客户的借款渠道。
   5. `is_extend`：是否展期，这个字段可以表示客户的还款行为。
   6. `owed_int_in`：表内欠息金额，这个字段可以表示客户的债务状况。
   7. `owed_int_out`：表外欠息金额，这个字段也可以表示客户的债务状况。
   8. `delay_bal`：逾期金额，这个字段可以表示客户的还款行为。

   ![image-20230516155946136](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537537.png)

3. djk_info '贷记卡开户明细'

   1. `is_withdrw`、`is_transfer`、`is_deposit`、`is_purchse`、`is_mob_bank`、`is_etc`: 这些字段表示用户的贷记卡功能开通情况，可以帮助我了解用户的信用卡使用情况。
   2. `cred_limit`: 信用额度，直接与credit挂钩，这是一个重要的信用信息。
   3. `deposit`、`over_draft`、`dlay_amt`: 这些字段分别表示贷记卡存款、普通额度透支和逾期金额，可以帮助我了解用户的贷款使用情况。
   4. `bal`: 贷记卡的开户余额对评估用户信用等级是一个重要的信用信息。

   ![image-20250307162118973](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071621025.png)

4. djkfq_info '贷记卡分期付款明细'

   1. `total_amt` - 总产品金额：能反映客户的贷款规模。
   2. `total_mths` - 总分期月数：能反映客户的还款周期。
   3. `instl_cnt` - 已分期摊销期数：能反映客户的还款进度。
   4. `rem_ppl` - 剩余未还本金：能反映客户的未偿还债务。
   5. `total_fee` - 总费用：能反映客户的总还款额度。
   6. `rem_fee` - 剩余未还费用：能反映客户的未偿还费用。

   ![image-20250307162036549](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071620601.png)

   ![image-20250307162044397](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071620438.png)

5. dm_v_tr_huanb_mx '贷款还本明细'

   统计每个用户的总还本金额。这是一个重要的指标，它可以反映出用户的偿还能力。可以将这个字段命名为 `total_repayment_amt`

   ![image-20230517130209216](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537104.png)

   ![image-20250307162009539](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071620574.png)

6. dm_v_tr_huanx_mx '贷款还息明细'

   同样，我可以将 '贷款还息明细' 表中的交易金额字段（这里是利息）进行求和操作，并根据每个用户的唯一标识 `uid` 进行聚合。这将给我每个用户的总还款利息。命名为`total_repayment_interest`

   ![image-20250307161925877](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071619706.png)

   ![image-20250307161959408](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071619443.png)

7. pri_cust_base_info '基本信息'

   合并这些`is_`开头的字段的原因是它们可能会提供关于客户的重要信息，如是否是员工、是否是股东、是否在黑名单中等等。这些信息可能会对预测客户的信用等级或者贷款偿还能力有所帮助。再者，这些字段都是二进制的，非常适合于大多数机器学习模型的输入。

   ![image-20230517124152245](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537486.png)

8. dm_v_tr_djk_mx '贷记卡交易

   ![image-20230516170333219](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537490.png)

   ![image-20230516170323032](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537498.png)

这些表格可以提供与客户贷款和贷记卡有关的信息，有助于评估客户的credit_level。

表格结构：

```
credit '客户信用信息'(
uid -- 证件号码
credit_level -- 客户信用等级
all_bal_loan -- 所有贷款的总余额，它可以反映客户的贷款负担
bad_bal_loan -- 不良贷款余额，它可以反映客户的信用状况
due_intr_loan -- 欠息总额，可以反映客户的贷款偿还情况
norm_bal_loan -- 正常贷款余额，可以反映客户的贷款结构
delay_bal_loan -- 逾期贷款总额，它可以反映客户的贷款偿还情况
num_of_accounts -- 账户数量
avg_loan_amt -- 平均贷款金额，一个重要的信贷特征
avg_loan_bal -- 平均贷款余额，也是一个重要的信贷特征
is_withdrw -- 是否进行过取款，可以帮助我了解用户的信用卡使用情况
is_transfer -- 是否进行过转账，可以帮助我了解用户的信用卡使用情况
is_deposit -- 是否进行过存款，可以帮助我了解用户的信用卡使用情况
is_purchse -- 是否进行过购买，可以帮助我了解用户的信用卡使用情况
is_mob_bank -- 是否使用过手机银行，可以帮助我了解用户的信用卡使用情况
is_etc -- 是否进行过其他交易，可以帮助我了解用户的信用卡使用情况
avg_cred_limit -- 平均信用额度，这是一个重要的信用信息
avg_deposit -- 平均存款金额，可以帮助我了解用户的贷款使用情况
avg_over_draft -- 平均透支金额，可以帮助我了解用户的贷款使用情况
avg_dlay_amt -- 平均逾期金额，可以帮助我了解用户的贷款使用情况
avg_bal -- 平均账户余额，这也是一个重要的信用信息
is_employee -- 是否是员工，这些信息可能会对预测客户的信用等级或者贷款偿还能力有所帮助
is_shareholder -- 是否是股东，这些信息可能会对预测客户的信用等级或者贷款偿还能力有所帮助
is_black -- 是否在黑名单中，这些信息可能会对预测客户的信用等级或者贷款偿还能力有所帮助
is_contact -- 是否是关联人，这些信息可能会对预测客户的信用等级或者贷款偿还能力有所帮助
is_mgr_dep -- 是否是营销部客户，这些信息可能会对预测客户的信用等级或者贷款偿还能力有所帮助
tran_count -- 交易次数
total_tran_amt -- 总交易金额
total_amt_sum -- 总产品金额，能反映客户的贷款规模
total_mths_sum -- 总分期月数，能反映客户的还款周期
instl_cnt_sum -- 已分期摊销期数，能反映客户的还款进度
rem_ppl_sum -- 剩余未还本金，能反映客户的未偿还债务
total_fee_sum -- 总费用，能反映客户的总还款额度
rem_fee_sum -- 剩余未还费用，能反映客户的未偿还费用
total_repayment_amt -- 总还本金额，这是一个重要的指标，它可以反映出用户的偿还能力
total_repayment_interest -- 总还款利息，每个用户的总还款利息
total_pprd_rfn_amt	--	贷款还本明细中的总每期还款金额
total_pprd_amotz_intr	--	贷款还本明细中的总每期摊还额计算利息
total_tran_amt	--	贷款还本明细中的总交易金额
last_bal	--	贷款还本明细中的最后余额
total_tran_amt_huanx	--	贷款还息明细中的总利息
avg_intr	--	贷款还息明细中的平均利率
total_cac_intc_pr	--	贷款还息明细中的总计息本金
)
```

### b. 数据盘点

描述数据集的分位数和频数，**使用可视化方式**。

1. 描述性统计信息分析：使用pandas的`describe`方法计算选中的数值列的描述性统计信息，包括计数、平均值、标准偏差、最小值、四分位数和最大值等，并将这些信息转置后存储为一个新的DataFrame，然后将此DataFrame导出为新的CSV文件，使用可视化工具进行各个属性的可视化，见img/数据盘点图片。

   ![image-20230521220742602](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537502.png)

   ![image-20230521220838559](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537912.png)

2. 频数统计和直方图创建：对于每个数值列，计算并打印出其频数，即每个值出现的次数。然后，使用Seaborn库的`histplot`方法创建直方图，以图形方式表示各数值的频数分布。在直方图上，每个柱体的高度表示相应数值的频数，柱体的宽度表示数值的区间范围。同时，代码还在每个柱体的顶部标注了其高度（即频数），最后，将直方图保存为PNG格式的图片。详情见Histogram for XXXX in Credit Dataset.png。

   ![image-20230521221545652](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537935.png)

## 2、数据预处理

### a. 数据标准化

数据标准化是数据预处理的重要环节。该过程旨在消除数据的量纲影响，让每个特征的分布具有一致性，这样能够让算法在学习过程中更好地理解数据，从而提高预测或分类的准确性。在这个代码中，使用了 Z-score 标准化方法。

Z-score 标准化是一种常用的标准化方式，也称为标准分数，是将数据按比例缩放，使之落入一个小的指定区间内。这种方式要求原始数据的分布可以近似为高斯分布，否则效果会很差。Z-score 标准化公式为：x = (x - μ) / σ，其中 μ 为所有样本数据的均值，σ 为所有样本数据的标准差。

关键代码片段：

```python
# 创建 StandardScaler 实例
scaler = StandardScaler()

# 选择要标准化的数值列，排除 'uid' 和 'star_level' 或 'credit_level'
numeric_columns = [col for col in df.columns if (df[col].dtypes in ['int64', 'float64']) and col not in ['uid', 'star_level', 'credit_level']]

# 将 Z-score 标准化应用到数据
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
```

### b. 缺失值处理

缺失值处理是数据预处理的重要步骤，我采用了随机森林算法对缺失值进行填充。随机森林是一种集成学习方法，可以处理分类或回归问题。当应用于缺失值填充时，随机森林可以基于数据的其他特征来预测缺失值。在本代码中，随机森林是用于预测每列中的缺失值，并用预测值填充缺失值。选择原因：

1. 适用性强
2. 易于处理高维特征
3. 缺点是有大量缺失值是判断容易出现失误，但我已经在缺失值处理前统计各列数据缺失率并剔除缺失值较大的列；缺点在当前数据环境中不易体现

关键代码片段：

```python
# 对每一列应用随机森林填充，除了不需要处理的列
for col in df.columns:
    # 如果某个列缺失值大于90%，剔除该列
    if df[col].isnull().mean() > 0.9:
        df = df.drop(col, axis=1)
    # 如果某个列缺失值不大于90%，使用随机森林进行填充
    elif col not in not_to_process and df[col].isnull().any():
        df = impute_with_rf(df, col)
        df_notnull = df.loc[df[column].notnull()]
    df_isnull = df.loc[df[column].isnull()]

    X = df_notnull[columns]
    y = df_notnull[column]

    rf = RandomForestRegressor(random_state=0, n_estimators=100)
    rf.fit(X, y)

    predicted_values = rf.predict(df_isnull[columns])
    df_isnull[column] = predicted_values

    df_new = pd.concat([df_notnull, df_isnull])

    return df_new
```

### c. 异常值处理

异常值处理也是数据预处理的重要步骤，因为异常值的存在会影响到数据分析的结果。在这个代码中，采用了 Z-score 方法进行异常值检测，然后用中位数进行替换。

Z-score 是一种常用的异常值检测方法，其原理是如果一个值的标准分数（即 Z-score）大于3或者小于-3，那么这个值就可以被视为异常值。在代码中，对每一列的数据进行了 Z-score 计算，并将计算结果大于3或者小于-3的值替换为这一列的中位数。

关键代码片段：

```python
# 异常值处理
# 通常情况下，我会选择3σ原则（或者说是Z-score方法）进行异常值检测，
# 这里简单起见，我直接把所有大于3个标准差的数值视为异常值，用该列的中位数进行替换
for column in df.columns:
    if column not in not_to_process and df[column].dtype != 'object':  # 仅处理数值型数据
        mean = df[column].mean()
        std = df[column].std()
        outliers = df[(df[column] - mean).abs() > 3 * std]
        df.loc[outliers.index, column] = df[column].median()
```


### d. 数据转换

需要注意的是，这一步我在clickhouse仓库合并时就**已经完成**，并体现在截图的SQL语句当中，包括：

- 数据去重，保证数量，uid与原credit_info/star_info相同
- 多行数据合并（例如统计一个人交易总额或平均交易额度，持有不同等级信用卡数量等）
- 将非数值特征编码成数值类型（多个is开头字段，对于1uid对应多个同is字段采用数字相加形式--一人的信用卡有几张开了ETC）
- 异常值校验（同一时间戳发生的交易有多个不同金额出现，仅选取一个作为参考）

## 3、特征工程与选择

### a. 特征工程

#### star_level

##### 计算相关性

已知star_level与客户存款有关，与客户存款相关的字段如下：

> all_bal_summary -- '存款汇总信息.总余额'，客户在银行所有账户的存款总余额
> avg_month_summary -- '存款汇总信息.月日均'，客户在银行所有账户的月日均存款余额
> avg_quarter_summary -- '存款汇总信息.季度日均'，客户在银行所有账户的季度日均存款余额
> avg_year_summary -- '存款汇总信息.年日均'，客户在银行所有账户的年日均存款余额
> current_deposit_summary -- '存款汇总信息.活期余额'，客户的所有活期存款余额
> fixed_deposit_summary -- '存款汇总信息.定期余额'，客户的所有定期存款余额
> financial_management_balance_summary -- '存款汇总信息.理财余额'，客户所有理财产品的余额
> avg_acct_bal_account -- '存款账号信息.账户余额均值'，客户所有账户的平均余额
> avg_bal_account -- '存款账号信息.原币余额均值'，客户所有账户的平均原币余额
> avg_avg_mth_account -- '存款账号信息.月日均均值'，客户所有账户的平均月日均存款余额
> avg_avg_qur_account -- '存款账号信息.季度日均均值'，客户所有账户的平均季度日均存款余额
> avg_avg_year_account -- '存款账号信息.年日均均值'，客户所有账户的平均年日均存款余额

考虑到进行相关性考察的双方均没有呈现正态分布，且star_level是枚举类型的非线性数据，选择斯皮尔曼系数进行特征工程。计算完相关性系数后还进行了显著性的判断（p=0.05）。

代码如下：

```python
import pandas as pd
from scipy import stats

if __name__ == '__main__':

    data = pd.read_csv('star_to_train.csv')

    # 计算相关系数并进行相关性检验
    for col in data.columns:
        if col != 'star_level':  # 排除star_level本身

            unique_data = data[[col, 'star_level']]

            # 单独处理特殊情况
            if len(unique_data[col].unique()) <= 1:
                print(f"{col}: 相关系数：NaN, p-value：NaN")
            else:
                coef, p_value = stats.spearmanr(unique_data[col], unique_data['star_level'])
                print(f"{col}: 相关系数：{coef}, p-value： {p_value}")

```

结果如下：

![QQ图片20230517151613](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537953.jpg)

全部字段的结果如下：

![QQ图片20230517161053](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537972.jpg)

p-value不存在为0的情况，可能是计算机计算浮点数精度不足的情况，此处考虑p-value为一非常接近0的值。

avg_quarter_summary为NaN是由于0值过多造成的。

首先剔除p-value大于0.05的情况，再参考资料，选择强相关性的字段（相关系数大于0.7），结果如下：

> all_bal_summary -- '存款汇总信息.总余额'，客户在银行所有账户的存款总余额
> avg_month_summary -- '存款汇总信息.月日均'，客户在银行所有账户的月日均存款余额
> avg_year_summary -- '存款汇总信息.年日均'，客户在银行所有账户的年日均存款余额
> fixed_deposit_summary -- '存款汇总信息.定期余额'，客户的所有定期存款余额

##### 计算VIF

考虑到这些字段之间可能存在较强的共线性，故多重共线性。

代码如下：

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == '__main__':

    data = pd.read_csv('star_to_train.csv')

    X = data[['all_bal_summary','avg_month_summary',
              'avg_year_summary','fixed_deposit_summary']]  # 选择要计算VIF的字段
    X['intercept'] = 1	# 添加常数列（截距项）

	# 计算vif
    vif = pd.DataFrame()
    vif["Features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif)
```

结果如下：

![QQ图片20230517160951](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537005.jpg)

#### credit_level

##### 计算相关性

> 字段名	类型	描述
> all_bal_loan	数值	客户所有贷款余额
> bad_bal_loan	数值	客户不良贷款余额
> due_intr_loan	数值	客户到期应收利息的贷款
> norm_bal_loan	数值	客户正常贷款余额
> delay_bal_loan	数值	客户逾期贷款余额
> num_of_accounts	数值	客户拥有的账户数量
> avg_loan_amt	数值	客户平均贷款金额
> avg_loan_bal	数值	客户贷款余额的平均值
> is_withdrw	字符串	是否进行过取款
> is_transfer	字符串	是否进行过转账
> is_deposit	字符串	是否进行过存款
> is_purchse	字符串	是否进行过购买
> is_mob_bank	字符串	是否使用过移动银行
> is_etc	字符串	是否使用过其他服务
> avg_cred_limit	数值	客户的平均信贷额度
> avg_deposit	数值	客户的平均存款金额
> avg_over_draft	数值	客户的平均透支金额
> avg_dlay_amt	数值	客户的平均逾期金额
> avg_bal	数值	客户的平均余额
> is_employee	字符串	是否是员工
> is_shareholder	字符串	是否是股东
> is_black	字符串	是否是黑名单
> is_contact	字符串	是否是联系人
> is_mgr_dep	字符串	是否是管理层
> tran_count	数值	交易数量
> total_tran_amt	数值	总交易金额
> total_amt_sum	数值	总金额和
> total_mths_sum	数值	总月数
> instl_cnt_sum	数值	总分期次数
> rem_ppl_sum	数值	总余额
> total_fee_sum	数值	总费用
> rem_fee_sum	数值	剩余费用
> total_repayment_amt	数值	总还款金额
> total_repayment_interest	数值	总还款利息
> total_pprd_rfn_amt	数值	贷款还本明细中的总每期还款金额
> total_pprd_amotz_intr	数值	贷款还本明细中的总每期摊还额计算利息
> total_tran_amt	数值	贷款还本明细中的总交易金额
> last_bal	数值	贷款还本明细中的最后余额
> total_tran_amt_huanx	数值	贷款还息明细中的总利息
> avg_intr	数值	贷款还息明细中的平均利率
> total_cac_intc_pr	数值	贷款还息明细中的总计息本金

考虑到进行相关性考察的双方均没有呈现正态分布，选择斯皮尔曼系数进行特征工程。计算完相关性系数后还进行了显著性的判断（p=0.05）。

代码同上。

结果如下：

![QQ图片20230519101003](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537381.jpg)

p-value不存在为0的情况，可能是计算机计算浮点数精度不足的情况，此处考虑p-value为一非常接近0的值。

avg_bal为NaN是由于0值过多造成的。

首先剔除p-value大于0.05的情况，再选择相关性相对较强的字段（相关系数绝对值大于0.3），结果如下：

> avg_loan_amt	数值	客户平均贷款金额
> avg_loan_bal	数值	客户贷款余额的平均值
> total_pprd_amotz_intr	数值	贷款还本明细中的总每期摊还额计算利息
> last_bal	数值	贷款还本明细中的最后余额
> total_tran_amt_huanx	数值	贷款还息明细中的总利息
> total_cac_intc_pr	数值	贷款还息明细中的总计息本金

##### 计算VIF

考虑到这些字段之间可能存在较强的共线性，故多重共线性。

代码同上。

结果如下：

![QQ图片20230519101954](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537424.jpg)

### b. 特征选择

对于特征选择，我采取了迭代的方式。即首先选取所有的特征，然后逐步剔除不重要的特征，每一步都要评估模型的性能，直到性能下降或者不能再删除特征为止。与此同时，还需要检查特征之间的关联性。如果两个特征高度相关，那么他们可能会在模型中提供重复的信息，增加模型的复杂性而没有增加太多的价值。在这种情况下，我可以选择保留其中一个特征，剔除另一个。最终，在参考特征工程给出信息的基础上，我逐步添加/剔除不同列，对比模型的平均性能（通过准确率，精确率，召回率等指标）我选择了下图的数据：（蓝色为特征工程推荐保留，作为我逐步迭代的出发点，红色则代表迭代中添加的列）

![QQ图片20230521231256](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537476.jpg)

![QQ图片20230521231304](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537502.png)

## 4. 模型选择、评估与应用

- 采用了多于一种模型并对不同模型进行评估
- 给出F1分数，准确率，精确率，召回率，混淆矩阵，Cohen's Kappa系数，以及其他的可以评估模型准确度的指标
- 对比各种模型并选择合适的模型（集成投票分类器）

我选择了逻辑回归、决策树、随机森林、XGBoost和集成投票分类器，其使用少数服从多数的hard voting。

### 4.1 逻辑回归

credit：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.61   | 0.43   | 0.61   | 0.48   | -0.0002   |

混淆矩阵：

![image-20230521232733977](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537916.png)

star：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.83   | 0.81   | 0.83   | 0.81   | 0.7054    |

混淆矩阵：

![image-20230521232018851](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537931.png)

### 4.2 决策树

credit：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.88   | 0.89   | 0.88   | 0.88   | 0.7641    |

混淆矩阵:

![image-20230521232718963](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537948.png)

star：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.78   | 0.71   | 0.78   | 0.73   | 0.5861    |

混淆矩阵：

![image-20230521232523865](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537962.png)

### 4.3 随机森林

credit：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.91   | 0.91   | 0.91   | 0.90   | 0.8174    |

混淆矩阵：

![image-20230521232754108](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537981.png)

star：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.92   | 0.92   | 0.92   | 0.92   | 0.8422    |

混淆矩阵：

![image-20230521232543287](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537157.png)

### 4.4 XGBoost

credit：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.90   | 0.92   | 0.91   | 0.91   | 0.8192    |

![image-20230521232654335](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537523.png)

star：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.91   | 0.90   | 0.92   | 0.92   | 0.8379    |

![image-20230521232502521](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537539.png)

### 4.5 VotingClassifier

credit：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.91   | 0.91   | 0.91   | 0.92   | 0.8552    |

![image-20230521232834032](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537605.png)

star：

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
| ------ | ------ | ------ | ------ | --------- |
| 0.92   | 0.91   | 0.92   | 0.92   | 0.8458    |

![image-20230521232605906](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537623.png)

### 4.6 关键代码

1. 根据特征工程的结果选择待训练的列

![image-20230519212536819](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537642.png)

2. 模型训练与预测：以votingClassifier为例，利用sklearn库集成3种模型，进行fit和predict

![image-20230519213323637](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537851.png)

3. 模型评估：将数据集化为80%的训练集和20%的测试集，分别输出准确率、kappa系数、混淆矩阵等，便于进行模型评估

![image-20230519213508434](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537869.png)

4. 模型应用：选择表现最优的xgboost，预测缺失的level。

![image-20230519213919703](https://lapsey-pictures.oss-cn-shenzhen.aliyuncs.com/typora_imgs/202503071537959.png)

### 4.7 总结

因为逻辑回归的某些指标不尽人意，在最后进行代码修改时，分类选择器中未加入逻辑回归模型。

其次，对于credit和star，随机森林、XGBoost、VotingClassifier均表现良好，准确率均大于90%，kappa系数在0.80-0.85之间，而VotingClassifier以略微优势胜出。所以在预测阶段，我选择了VotingClassifier模型得出credit_level和star_level，将结果写入credit_res.csv与star_res.csv当中。
