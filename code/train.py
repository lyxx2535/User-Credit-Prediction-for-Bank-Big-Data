import csv
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Specify the file type
# file_type = 'credit'
file_type = 'star'


def to_of_star(before_trains):
    # Convert star levels to a list of integers by subtracting 1
    a = []
    for item in before_trains:
        a.append(int(item) - 1)
    return a


def from_of_star(before_trains):
    # Convert star levels back to the original form by adding 1
    a = []
    for item in before_trains:
        a.append(int(item) + 1)
    return a


def write_predict_data_to_csv(path, xgboost_y_predict):
    # Read the data from the CSV file
    with open(path, mode='r', encoding='utf-8') as file:
        data = list(csv.reader(file))

    if file_type == 'star':
        # Create a new data list with the predicted data in the last column
        new_data = [['uid', 'star_level']]
        for i in range(1, len(data)):
            new_data.append([data[i][0], int(xgboost_y_predict[i - 1])])
        # Write the new data to a CSV file for star predictions
        with open('./data/star_res.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_data)
    else:
        # Create a new data list with the predicted data in the last column
        new_data = [['uid', 'credit_level']]
        for i in range(1, len(data)):
            new_data.append([data[i][0], int(xgboost_y_predict[i - 1])])
        # Write the new data to a CSV file for credit predictions
        with open('./data/credit_res.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_data)


def get_data_from_path(path):
    # Read the original data from the file
    X = []
    y = []
    csvFile = open(path, "r", encoding='utf-8')
    reader = csv.reader(csvFile)
    next(reader)

    for item in reader:
        # del item[0]
        if file_type == 'star':
            # Select relevant columns for star predictions
            last = item.pop()
            item = [item[2], item[3], item[5], item[7], last]
        if file_type == 'credit':
            # Select relevant columns for credit predictions
            last = item.pop()
            item = [item[7], item[8], item[36], item[37], item[38], item[40], last]
        if item.__contains__(""):
            continue
        item = [float(ii) for ii in item]
        X.append(item)

    # Convert the data to float format
    for i in range(len(X)):
        y.append(X[i].pop())
    return X, y


def to_of_credit(before_trans):
    # Convert credit levels to a list of integers
    a = []
    for item in before_trans:
        if item == 35:
            a.append(0)
        elif item == 50:
            a.append(1)
        elif item == 60:
            a.append(2)
        elif item == 70:
            a.append(3)
        elif item == 85:
            a.append(4)
    return a


def from_of_credit(before_trans):
    # Convert credit levels back to the original form
    a = []
    for item in before_trans:
        if item == 0:
            a.append(35.0)
        elif item == 1:
            a.append(50.0)
        elif item == 2:
            a.append(60.0)
        elif item == 3:
            a.append(70.0)
        elif item == 4:
            a.append(85.0)
    return a


def logic_regression_train_and_predict(X_train, X_test, y_train):
    # Train a logistic regression model and make predictions
    lr_clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='saga', C=10.0)
    lr_clf = lr_clf.fit(X_train, y_train)
    logistic_regression_ans = lr_clf.predict(X_test)
    return logistic_regression_ans


def decision_tree_train_and_predict(X_train, X_test, y_train):
    # Train a decision tree model and make predictions
    clf = DecisionTreeClassifier(max_depth=30, min_samples_leaf=3, min_impurity_decrease=0.1)
    clf = clf.fit(X_train, y_train)
    decision_tree_ans = clf.predict(X_test)
    return decision_tree_ans


def random_forest_train_and_predict(X_train, X_test, y_train):
    # Train a random forest model and make predictions
    rfc = RandomForestClassifier(class_weight='balanced', random_state=37)
    rfc = rfc.fit(X_train, y_train)
    return rfc.predict(X_test)


def xgboost_train_and_predict(X_train, X_test, y_train):
    # Train an XGBoost model and make predictions
    model = xgb.XGBClassifier(booster='gbtree', objective='multi:softmax', num_class=3, gamma=0.1, max_depth=6,
                              subsample=0.7, colsample_bytree=0.7, min_child_weight=3, slient=1, eta=0.1)

    if file_type == 'star':
        model.fit(X_train, to_of_star(y_train))
        ans = model.predict(X_test)
        return from_of_star(ans)
    else:
        model.fit(X_train, to_of_credit(y_train))
        ans = model.predict(X_test)
        return from_of_credit(ans)


def vote_train_and_predict(X_train, X_test, y_train):
    # Train a voting classifier model and make predictions
    lr_clf = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=100000)
    dtc_clf = DecisionTreeClassifier(max_depth=30, min_samples_leaf=3, min_impurity_decrease=0.1)
    rfc_clf = RandomForestClassifier(class_weight='balanced', random_state=37)
    xgb_clf = xgb.XGBClassifier(booster='gbtree', objective='multi:softmax', num_class=3, gamma=0.1, max_depth=6,
                                subsample=0.7, colsample_bytree=0.7, min_child_weight=3, slient=1, eta=0.1)
    vote_clf = VotingClassifier(
        estimators=[('xgboost', xgb_clf), ('logistic_regression', lr_clf), ('decision_tree', dtc_clf),
                    ('random_forest', rfc_clf)], voting='hard')
    if file_type == 'star':
        vote_clf.fit(X_train, to_of_star(y_train))
        return from_of_star(vote_clf.predict(X_test))
    else:
        vote_clf.fit(X_train, to_of_credit(y_train))
        return from_of_credit(vote_clf.predict(X_test))


def score_model_read(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

    # Use various models to make predictions and evaluate their performance
    xgboost_y_predict = xgboost_train_and_predict(X_train, X_test, y_train)
    logic_regression_y_predict = logic_regression_train_and_predict(X_train, X_test, y_train)
    decision_tree_y_predict = decision_tree_train_and_predict(X_train, X_test, y_train)
    random_forest_y_predict = random_forest_train_and_predict(X_train, X_test, y_train)
    vote_y_predict = vote_train_and_predict(X_train, X_test, y_train)

    # Print classification reports for each model
    print(classification_report(y_test, logic_regression_y_predict))
    print(classification_report(y_test, decision_tree_y_predict))
    print(classification_report(y_test, random_forest_y_predict))
    print(classification_report(y_test, xgboost_y_predict))
    print(classification_report(y_test, vote_y_predict))

    # Print Cohen's kappa scores for each model
    print(cohen_kappa_score(y_test, logic_regression_y_predict))
    print(cohen_kappa_score(y_test, decision_tree_y_predict))
    print(cohen_kappa_score(y_test, random_forest_y_predict))
    print(cohen_kappa_score(y_test, xgboost_y_predict))
    print(cohen_kappa_score(y_test, vote_y_predict))

    # Plot confusion matrices
    cm1 = confusion_matrix(y_true=y_test, y_pred=logic_regression_y_predict)
    cm2 = confusion_matrix(y_true=y_test, y_pred=decision_tree_y_predict)
    cm3 = confusion_matrix(y_true=y_test, y_pred=random_forest_y_predict)
    cm4 = confusion_matrix(y_true=y_test, y_pred=xgboost_y_predict)
    cm5 = confusion_matrix(y_true=y_test, y_pred=vote_y_predict)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    # for i, cm in enumerate([cm1, cm2, cm3,cm5]):
    for i, cm in enumerate([cm1, cm2, cm3, cm4, cm5]):
        ax[i].imshow(cm, cmap=plt.cm.Blues)
        ax[i].set_title('Confusion Matrix {}'.format(i + 1))
        ax[i].set_xlabel('Predicted label')
        ax[i].set_ylabel('True label')
        tick_marks = np.arange(len(set(y_test)))
        ax[i].set_xticks(tick_marks)
        ax[i].set_xticklabels(list(set(y_test)))
        ax[i].set_yticks(tick_marks)
        ax[i].set_yticklabels(list(set(y_test)))
        for j in range(len(set(y_test))):
            for k in range(len(set(y_test))):
                ax[i].text(k, j, format(cm[j][k], 'd'), ha="center", va="center",
                           color="white" if cm[j][k] > cm.max() / 2. else "black")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if file_type == 'star':
        # Read preprocessed training and test data for star predictions
        X_train, y_train = get_data_from_path("./data/star_to_train.csv")
        X_test, y_test = get_data_from_path("./data/star_to_fill.csv")

        # Score the models and make predictions
        score_model_read(X_train, y_train)
        # xgboost_y_predict = xgboost_train_and_predict(X_train, X_test, y_train)
        vote_y_predict = vote_train_and_predict(X_train, X_test, y_train)
        # Write the predicted data to a CSV file for star predictions
        # write_predict_data_to_csv('./data/star_to_fill.csv', xgboost_y_predict)
        write_predict_data_to_csv('./data/star_to_fill.csv', vote_y_predict)
    else:
        # Read preprocessed training and test data for credit predictions
        X_train, y_train = get_data_from_path("./data/credit_to_train.csv")
        X_test, y_test = get_data_from_path("./data/credit_to_fill.csv")

        # Score the models and make predictions
        score_model_read(X_train, y_train)
        # xgboost_y_predict = xgboost_train_and_predict(X_train, X_test, y_train)
        vote_y_predict = vote_train_and_predict(X_train, X_test, y_train)
        # Write the predicted data to a CSV file for credit predictions
        write_predict_data_to_csv('./data/credit_to_fill.csv', vote_y_predict)
        # write_predict_data_to_csv('./data/credit_to_fill.csv', xgboost_y_predict)
