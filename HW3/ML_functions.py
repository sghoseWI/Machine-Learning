import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import graphviz
import math
import seaborn as sns
get_ipython().magic('matplotlib inline')


def read_data(filepath, file_type = None):
    if file_type == 'csv':
        full_df = pd.read_csv(filepath)
    elif file_type == 'excel':
        full_df = pd.read_excel(filepath)
    elif file_type == 'json':
        full_df = pd.read_json(filepath)
    return full_df

def add_dummy_variable(df, var, dummy_var, lambda_equation):
    df[dummy_var] = df[var].apply(lambda_equation)

def test_train_data_split(df, var, test_size):
    X = df
    Y = df[var]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def pre_process(df, var):
    return df['var'].describe()

def convert_true_false_1_0(df):
    col_list = list(df.columns)
    for col in col_list:
        if 'f' in list(df[col].unique()):
            df[col] = df[col].apply(lambda x: 1 if x=='t' else 0)

def find_missing_values(df):
    missing_df = df.isnull().sum().sort_values(ascending = False)
    return missing_df

def impute_missing_values(df, var, fill_method = None):
    if fill_method == 'mean':
        df[var] = df[var].fillna(df[var].mean())
    if fill_method == 'mode':
        df[var] = df[var].fillna(df[var].mode())
    if fill_method == 'median':
        df[var] = df[var].fillna(df[var].median())

def visualize_correlation_heatmap(df):
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    plt.show()

def visualize_line_graph(df, var, title, xlabel, ylabel):
    x = list(df.index.values)
    y = list(df[var].values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.show()

def visualize_bar_chart(xvals, yvals, xlabel, ylabel, title, width = 0.35, color = 'blue'):
    counts = yvals

    ind = np.arange(len(xvals))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, counts, width, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xvals)

    plt.gcf().subplots_adjust(bottom=0.2)

    plt.show()

def LR():
    return LogisticRegression(penalty = 'l1', C = 1e5)

def KNN():
    return KNeighborsClassifier(n_neighbors = 3)

def DT():
    return DecisionTreeClassifier()

def SVM():
    return svm.SVC(kernel = 'linear', probability = True, random_state = 3)

def RF():
    return RandomForestClassifier(n_estimators = 50, n_jobs = -1)

def AB():
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                    algorithm="SAMME",
                                                    n_estimators=200)

def GB():
    return GradientBoostingClassifier(learning_rate = 0.05,
                                    	subsample = 0.5,
                                    	max_depth = 6,
                                    	n_estimators = 10)
def NB():
    return GaussianNB()

def classifier_accuracy_score(classifiers_list, X_train, Y_train, X_output, Y_output):
    classifier_and_score_list = []
    for classifier in classifiers_list:
        accuracy_score = classifier.score(X_output, Y_output)
        classifier_and_score_list.append((classifier, accuracy_score))
    return classifier_and_score_list

def precision_recall_threshold(y_test, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds

def plot_precision_recall_binary_classification(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
