#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:34:50 2018
HOMEWORK 2: ML Pipeline
@author: elenabg
"""
import sys
import time
import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive
import seaborn as sns
import csv
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pylab as pl
from sklearn.metrics import accuracy_score as accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


##1. Read dataset

def load_data(path, filename, format_ = 'csv', dups = False):
    '''
    Reads data from an external source into pandas.
    
    Inputs:
    - path (str) 
    - filename (str)
    - format_ (str): csv (default), json, stata, excel
    - dups (bool): False (default)
    
    Returns:
    - pandas.DataFrame with full dataset
    - pandas.DataFrame with duplicate rows, if dups is set to True
    '''
    if format_ == 'csv':
        df_all = pd.read_csv(path + filename)
    elif format_ == 'json':
        df_all = pd.read_json(path + filename)
    elif format_ == 'stata':
        df_all = pd.read_stata(path + filename, convert_categoricals = False)
    elif format_ == 'excel':
        df_all = pd.read_excel(path + filename)

    if dups:
        df_all['Dup'] = df_all.duplicated(subset= df_all.columns, keep = False)
        df_dups = df_all[df_all.Dup == True]
        df_all = df_all.drop(labels=['Dup'], axis=1)
        return df_all, df_dups
    else:
        return df_all


## 2. Explore Data

# 2.1 Dimensions and types

def overview(df, df_dups=None):
    '''
    Shows the number of rows and columns in the full dataframe and the duplicate rows
    dataframe, and each variable type in the full dataframe
    '''
    if df_dups is not None:
        dp = 'DUPLICATE ROWS:' + str(df_dups.shape[0])
    else:
        dp = ''
    print('DATASET DIMENSIONS: ' + str(df.shape[0]) + ' rows' ', ' + str(df.shape[1]) + \
          ' columns'  + '\n' + dp)
    return df.dtypes

def rename_cols(df, col_list, new_names):
    '''
    Inputs:
    - df (pd.DataFrame)
    - col_list (list of int): list of indexes
    - new_names (list of str): list of new columns names
    
    Returns: dataframe with renamed columns
    '''
    cols = [df.columns[i] for i in col_list]
    new_cols = dict(zip(cols, new_names))
    df.rename(columns = new_cols, inplace = True)

def to_datetime(df, var_list):
    for var in var_list:
        df[var] = pd.to_datetime(df[var],  errors='coerce')

def parse_date(td, units = 'D'):
    if units == 'Y':
        N = 364.0
    elif units == 'M':
        N = 30.0
    else:
        N = 1
    return float(td.days)/N

# 2.1  Distributions

def cnt_dist_elem(df, var):
    return len(df[var].unique())

def percents(df, factor):
    # how much of each user type do we have
    for name, g in df.groupby([factor]):
        print("{:.3f}% {} {}".format(100.0 * len(g) / len(df), factor, name))

def cond_stats(df, vars_interest, factor):
    '''
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of strings): list of column names
    - factor (str): column name of variable to group by
            
    Returns: table with conditional aggregate statistics
    '''
    
    df_subs = df[vars_interest]
    stats = df_subs.groupby(factor).agg([np.min, np.mean, np.median, np.max])
    return stats

def plot_distrib(df, var, title, bins = 25, cap=None):
    '''
    
    Inputs:
    - var (np.array)
    '''
    var = df.apply(lambda x: x.fillna(x.median()))[var]
    var.plot.hist(bins, alpha=0.3)
    plt.title(title)

def plot_mg_density(df, var, factor, title):
    '''
    Plot fitted kernel density to a variable ditribution, conditional on other variable.
    
    Inputs:
    - df (pd.DataFrame)
    - var (string)
    - factor (str)

    '''
    if factor:
        df.groupby(factor)[var].plot.kde()
        plt.title(title)
        plt.legend()
    else:
        df[var].plot.kde()
        plt.title(title)

# 2.2 Time Series 

def monthly_ts(df, month_var, title):
    df_gmth = pd.DataFrame({'Count' : df.groupby(['mnth_yr']).size()}).reset_index()
    ax = df_gmth.plot()
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks([i*5.5 for i in range(15)],
           ["Jan-00", "Mar-00", "Sep-00", "Dic-00", "Jan-02", "Mar-02", "Sep-02", "Dic-02","Jan-04", 
            "Mar-04", "Sep-04", "Dic-04", "Jan-06", "Mar-06", "Sep-06", "Dic-06"], rotation=90)
    plt.title(title)

# 2.3 Correlations

def multi_scatter(df, vars_interest):
    '''
    Plots a scatter for each pair of selected variables, with fitted kernel densities
    per variable in the main diagonal.
    
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of strings): list of column names
    '''
    df_subs = df[vars_interest]
    pd.plotting.scatter_matrix(df_subs, alpha = 0.7, figsize = (14,8), diagonal = 'kde')

def top_corr(df, vars_interest, label, heatmap = True):
    '''
    Computes pairwise correlation betweem label variable and selected
    features.
    Optionally, plots a heamap for each pair of correlations.
    
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of str): list of column names
    - label (str)

    Returns:
    - pd. Dataframe with sorted correlations
    '''
    df_subs = df[vars_interest]
    corr = df_subs.corr()
    corr_sort = df_subs.corr()[label].sort_values(ascending = False).to_frame()
    rename_cols(corr_sort, [0], ['correlation w/ delinquency'])
    if heatmap:
        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 20, as_cmap=True))
        #plt.title('Pairwise correlation between selected variables')
        
    return corr_sort


# 2.4 Outliers

def find_outliers(df, vars_interest, thresh):
    '''
    Detect outliers for each variable in the dataset, for a given threshold (measured in std dev
    from the mean)

    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of str)
    - thresh (float): minimum standard deviations from the mean to be considered an outlier

    Returns:
    - dict with variable as key and a list containg each outlier index
      and deviation magnitude as val.
    '''
    out_lst = {}
    df_subs = df[vars_interest]
    for var in df_subs.columns:
        mean = df[var].mean()
        std = df[var].std()
        
        for i, row in df_subs.iterrows():
            dev = abs((df_subs[var].loc[i] - mean) / std )
            if dev > thresh:
                if var not in out_lst:
                    out_lst[var] = [("MEAN:" + str(mean), "STD:" + str(std)), ("OUTLIER", "INDEX: " ,i, "DEV: ", dev)]    
                else:
                    out_lst[var].append(( "OUTLIER", "INDEX: " , i, "DEV: ", dev))
    return  out_lst
                

# 3. Pre-Processing

# 3.1 Impute Missing Values

def standarize(var):
    return (var - var.mean()) / var.std()

def simple_impute(df, val = 'median'):
    '''
    Fill in missing observations with a given value
    - val: {'med', 'mean', 0}
    '''
    if val == 'med':
        df = df.apply(lambda x: x.fillna(x.median()))
    elif val == 'mean':
        df = df.apply(lambda x: x.fillna(x.mean()))
    else:
        df = df.apply(lambda x: x.fillna(0))
    return df

# 4. Generate Features/Predictors

def daytime_vals(df, date_label, tvals):
    '''
    Extract time information from a datetime variable
    - df (pd.DataFrame)
    - date_label (str)
    - tvals (lst) {d, m, h, wd}
    '''
    
    if 'd' in tvals:
        df['day'] = df[date_label].apply(lambda t: t.day)
    if 'm' in tvals:
        df['month'] = df[date_label].apply(lambda t: t.month)
    if 'h' in tvals:
        df['hour'] = df[date_label].apply(lambda t: t.hour)
    if 'wd' in tvals:
        df['day_of_week'] = df[date_label].dt.weekday_name
        
# 4.1 Discretization and Cathegorization

def cap_values(x, cap_min, cap_max):
    '''
    Cap a value with a given max value.
    '''
    if x > cap_max or x < cap_min:
        return cap
    else:
        return x   

def categorize(df, vars_to_cat, bins, vars_to_cap=None):
    '''
    Build evenly spaced buckets for selected continous variables in a dataframe,
    cathegorize all selected variables with dummies, and add the new categorical
    variables to a dataframe.
    '''
    lst , lst_d = [], []
    for i, var in enumerate(vars_to_cat):
        name = var + '_cat'
        lst.append(name)
        name_d = var + '_dum'
        lst_d.append(name_d)
        
        if var in vars_to_cap:
            df[var] = df[var].apply(lambda x: cap_values(x, df[var].quantile(.05), df[var].quantile(.95)))
            col = pd.cut(df[var], bins[i])
            df[name] = col
            df[name_d] = df[name].cat.codes
        else:
            col = pd.cut(df[var], bins[i])
            df[name] = col
            df[name_d] = df[name].cat.codes

# 4.2 Visualize Features
    
def count_by(df, var, title, axis_labs):
    '''
    Compute and plot the number of ocurrences in the data, grouped by
    the possible values of a variable.
    
    df (pd.Dataframe)
    var (str)
    title (str)
    axis_labs (lst): x and y axis labels (str)
    
    '''
    pd.value_counts(df[var]).plot(kind='bar')
    plt.title(title)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])
    
    
def plot_cond_mean(df, var, label, title, axis_labs):
    '''
    Compute and plot the mean of a feature, grouped by
    the label possible values.
    '''
    plt.figure(1)
    df[[var, label]].groupby(var).mean().plot(legend=False, kind='bar')
    plt.title(title)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])
    
    plt.show()
        
def plot_cond_dist(df, factor, var, axis_labs, dens=False):
    '''
    Fit and plot a kernel density to the distribution of the label,
    grouped by the possible discretized values of a feature.
    '''

    plt.figure(2)
    if dens:
        df.groupby(factor)[var].plot.kde()
    else:
        df.groupby(factor)[var].plot.hist(bins=25)
    plt.title(title)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])
    plt.show()
    
    
def graph_by_fact(df, title, factor, var, axis_labs):
    '''
    Count and plot number of ocurrences in the data as per some variable,
    grouped by the possible values of a categorical feature.
    '''
    colors = plt.cm.GnBu(np.linspace(0, 1, len(df[factor].unique())))
    df_gmth = pd.DataFrame({'Count' : df.groupby([var, factor]).size()}).reset_index()
    #df_gmth = df_gmth[df_gmth.Count >=0]
    df_gmth_piv = df_gmth.pivot(index=var, columns=factor, values='Count')
    df_gmth_piv.plot(kind='bar', stacked=True, color = colors)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])
    plt.title(title)
    
def spatial_scatter(df,factor, lat_lab, lon_lab):
    '''
    
    factor (str)
    lat_lab : label for latitude (str)
    lat_lab : label for longitude (str)
    
    '''
        
    groups = df.groupby(factor)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.margins(0.05)
    for name, group in groups:
        if name == df[factor].unique()[0]:
            ax.plot(group[lat_lab], group[lon_lab], marker='o', linestyle='', ms=3, \
                    label=name, alpha = 1)
        elif name == df[factor].unique()[1]:
            ax.plot(group[lat_lab], group[lon_lab], marker='o', linestyle='', ms=2, \
            label=name, alpha = 0.3)
        else:
            ax.plot(group[lat_lab], group[lon_lab], marker='o', linestyle='', ms=2, \
            label=name, alpha = 0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.legend(fontsize=16)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=1.)
    plt.title("Spatial Distribution by {}".format(factor), fontsize=16)
    
def plot_importances(df, features, label):
    '''
    Build a random forest classifier to
    compute the relative importance of selected features in
    predicting the label.
    '''
    clf = RandomForestClassifier()
    clf.fit(df[features], df[label])
    importances = clf.feature_importances_
    np_features = np.array(features)
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(np_features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, np_features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")
    pl.show()

# 5. Build Classifier

def split_dataset(df, label, test_size = 0.3,):
    X = df.drop([label], axis=1) # predictors
    Y = df[label] # label
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = 42)
    return x_train, x_test, y_train, y_test

def fit_classifier(model, param, x_train, y_train):
    if model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=param)
    elif model == 'DT':
        clf = DecisionTreeClassifier(criterion='gini', max_depth=param)
    elif model == 'LR':
        clf = LogisticRegression('l2', C=param)
    clf.fit(x_train, y_train)
    return clf
        

# 6. Evaluate Classifier

def get_acc(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    tot = len(y_test)
    good = [i for i in range(len(y_pred)) if y_pred[i] == y_test.iloc[i]]
    acc = len(good) / tot
    return acc

def precision(y_pred, y_true):
    TP, FP = [], []
    for i, y in enumerate(y_pred):
        if y==1 and y_true[i] == 1:
            TP.append(i)
        elif y==1 and y_true[i] == 0:
            FP.append(i)
    prec = len(TP) / (len(TP) + len(FP))
    return prec

def recall(y_pred, y_true):
    TP, FN = [], []
    for i, y in enumerate(y_pred):
        if y==1 and y_true[i] == 1:
            TP.append(i)
        elif y==0 and y_true[i] == 1:
            FN.append(i)
    rec = len(TP) / (len(TP) + len(FN))
    return rec

def sample_cross_val(mod, param_lst, X_train, y_train, eval_metric):
    if mod == 'RF':
        clf = RandomForestClassifier()
        grid_values = {'n_estimators': param_lst}
    elif mod == 'LR':
        clf = LogisticRegression()
        grid_values = {'penalty': ['l1','l2'], 'C': param_lst}
    else:
        print('Enter RF or LR')
        return None
    
    grid = GridSearchCV(clf, grid_values, eval_metric, return_train_score = False, cv=5)
    grid.fit(X_train, y_train)
    return grid
