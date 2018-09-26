import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product
import math
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
import graphviz
import psycopg2
from __future__ import division
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
from scipy import optimize
import time
import seaborn as sns
import ML_functions as mlf
from sklearn.metrics import recall_score, f1_score, precision_score, roc_auc_score

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

#import get_census
import pickle
import fancyimpute
import warnings
import dotenv



warnings.filterwarnings('ignore')

# ssh variables
# host = 'postgres.dssg.io'
# localhost = 'mlclass.dssg.io'
# ssh_username = 'sghosewi'
# ssh_private_key = '~/.ssh/sghosewi'

# database variables
# user='jocodssg_students'
# password='aibaecighoobeeba'
# database='jocodssg'
CONNECTION_STRING = dotenv.get_key(dotenv.find_dotenv(), 'LCH_CONN')

# conn_string = "host='127.0.0.1' port='3308' dbname='jocodssg' user='jocodssg_students' password='aibaecighoobeeba'"
conn = psycopg2.connect(CONNECTION_STRING)
# conn = psycopg2.connect(host='127.0.0.1', port='3333', database='jocodssg', user='jocodssg_students', password='aibaecighoobeeba')
cur = conn.cursor()

def sql_to_df(q):
    with SSHTunnelForwarder(
        (host, 3333),
        ssh_username=ssh_username,
        ssh_private_key=ssh_private_key,
        remote_bind_address=(localhost, 22)
    ) as server:
        conn = psycopg2.connect(host=localhost,
                               port=server.local_bind_port,
                               user=user,
                               passwd=password,
                               db=database)

        return pd.read_sql_query(q, conn)

def run_query(sql_query, conn):
    try:
        cur = conn.cursor()
        cur.execute(sql_query)
        column_names = [desc[0] for desc in cur.description]
        res = cur.fetchall()
        res_named = [tuple(column_names)] + res
        cur.close()
        res_df = pd.DataFrame(res)
        res_df.columns = column_names
        return res_df
    except psycopg2.Error as e:
        print(e)
        conn.rollback()

mental_health_query='''
with mental_health as (
    select mh.dedupe_id,
    mh.patid,
    DATE_PART('day', mh.dschrg_date::timestamp - mh.admit_date::timestamp)::int as mh_stay_days,
    mh.dob,
    mh.program,
    mh.pri_dx_value,
    disch.admit_date,
    dschg_date,
    discharge_reason,
    diag.dx_description,
    diag.dx_date
    from clean.jocomentalhealth_hashed mh
    left outer join clean.jocomentalhealthdiagnoses diag
    on mh.patid = diag.patid
    left outer join clean.jocomentalhealthdischarges disch
    on diag.patid = disch.patid),
person_info as (
    select indiv.dedupe_id,
    indiv.sex,
    indiv.dob,
    extract(year from age(timestamp '2010-01-01', indiv.dob::timestamp))::int as age_calced,
    indiv.race,
    inm.mar_stat,
    inm.city,
    inm.zip,
    inm.arr_agency,
    inm.arrest_dt,
    inm.rel_date,
    extract(year from age(inm.arrest_dt::timestamp, inm.dob::timestamp))::int as arrest_age,
    inm.bk_dt,
    DATE_PART('day', inm.bk_dt::timestamp - inm.arrest_dt::timestamp)::int as unbooked_days,
    DATE_PART('day', inm.rel_date::timestamp - inm.bk_dt::timestamp)::int as stay_days
    from clean.individuals indiv left outer join clean.jocojims2inmatedata inm on indiv.dedupe_id = inm.dedupe_id
    )
select person.*,
mht.patid,
mht.mh_stay_days,
mht.program,
CASE WHEN mht.pri_dx_value IS NOT NULL
THEN 1
ELSE 0
END as mh_diagnosis,
mht.admit_date,
mht.dschg_date,
mht.discharge_reason,
mht.dx_description,
mht.dx_date
from person_info person left outer join mental_health mht
on person.dedupe_id = mht.dedupe_id and person.dob::timestamp = mht.dob::timestamp
'''

mental_health_query += ';'
mh_df = run_query(mental_health_query, conn)


RELEVANT_DF = mh_df[['returner',
   'anxiety_dummy', 'depression_dummy', 'psychotic_dummy', 'ptsd_dummy',
   'opps_dummy', 'drugs_dummy']]


def LR():
    return LogisticRegression(penalty = 'l1', C = 1e5)

def KNN():
    return KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)

def DT():
    return DecisionTreeClassifier()

def SVM():
    return svm.SVC(kernel = 'linear', probability = True, random_state = 3, n_jobs = -1)

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

NOTEBOOK = 0

def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    large_grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    small_grid = {
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    test_grid = {
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def scores_at_k(y_true, y_scores, k):
    '''
    Calculate precision, recall, and f1 score at a given threshold
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    return precision, recall, f1


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

def visualize_best_tree(best_tree, training_predictors, label_names = ['negative', 'positive'], filename = 'tree.dot'):
    '''
    Visualize decision tree object with GraphWiz
    '''
    viz = sklearn.tree.export_graphviz(best_tree,
                    feature_names = training_predictors.columns,
                    class_names=label_names,
                    rounded=False, filled=True)

    with open(filename) as f:
        dot_graph = f.read()
        graph = graphviz.Source(dot_graph)

    return graph


def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, thresholds = [5, 10, 20]):
    """
    Runs the loop using models_to_run, clfs, gridm and the data
    """
    # results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','baseline', 'p_at_1','p_at_2','p_at_5', 'p_at_10', 'p_at_20','p_at_30','p_at_50','recall_at_1','recall_at_2','recall_at_5','recall_at_10','recall_at_20','recall_at_30','recall_at_50','F1_at_1','F1_at_2','F1_at_5','F1_at_10','F1_at_20','F1_at_30','F1_at_50'))

    result_cols = ['model_type','clf', 'parameters', 'baseline_p', 'auc-roc']

    # define columns for metrics at each threshold specified in function call
    result_cols += list(chain.from_iterable(('p_at_{}'.format(threshold), 
                        'r_at_{}'.format(threshold), 
                        'f1_at_{}'.format(threshold)) for threshold in thresholds))

    results_df =  pd.DataFrame(columns=result_cols)

    TREE_COUNTER = 0

    for n in range(1, 2):
        # create training and valdation sets
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    # results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                      #  roc_auc_score(y_test, y_pred_probs),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,100.0),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                      #  precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                      #  recall_at_k(y_test_sorted,y_pred_probs_sorted, 1.0),
                                                      #  recall_at_k(y_test_sorted,y_pred_probs_sorted, 2.0),
                                                      #  recall_at_k(y_test_sorted,y_pred_probs_sorted, 5.0),
                                                      #  recall_at_k(y_test_sorted,y_pred_probs_sorted, 10.0),
                                                      #  recall_at_k(y_test_sorted,y_pred_probs_sorted, 20.0),
                                                      #  recall_at_k(y_test_sorted,y_pred_probs_sorted, 30.0),
                                                      #  recall_at_k(y_test_sorted,y_pred_probs_sorted, 50.0),
                                                      #  2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 1.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 1.0)),
                                                      #  2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 2.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 2.0)),
                                                      # 2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 5.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 5.0)),
                                                      # 2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 10.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 10.0)),
                                                      # 2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 20.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 20.0)),
                                                      # 2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 30.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 30.0)),
                                                      # 2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 50.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 50.0))]
                
                    results_list = [models_to_run[index], clf, p, 
                    precision_at_k(y_test_sorted, y_pred_probs_sorted, 100.0),
                    roc_auc_score(y_test, y_pred_probs)]

                    for threshold in thresholds:
                        precision, recall, f1 = mp.scores_at_k(y_test_sorted, 
                            y_pred_probs_sorted, threshold)
                        results_list += [precision, recall, f1]

                    results_df.loc[len(results_df)] = results_list



                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)

                        if models_to_run[index] == 'DT':
                            title = 'tree{}.dot'(TREE_COUNTER)
                            TREE_COUNTER += 1
                            visualize_best_tree(clf, X_train, label_names = ['Lower Risk', 'High Risk'], filename = title)

                except IndexError as e:
                    print('Error:',e)
                    continue

    return results_df

def count_previous_mh_services(df):
    count_df = mh_df.groupby('dedupe_id').dedupe_id.nunique()
    return count_df

def mh_to_cat(x, words_list):
    r = 0
    for word in words_list:
        if word in str(x):
            r = 1
    return r

def turn_to_1_0(df, b):
    for col in b:
        df[col] = df[col].apply(lambda x: 1 if x in mandatory_counseling_list else 0)

def add_treatment_complete_and_death_dummies(df):
    death_list = ['DEATH (NATURAL CAUSES-999.10)',
     'DEATH (SUICIDE-999.20)',
     'DEATH (UNKNOWN-999.60)',
     'DEATH (SUICIDE-999.20)        ',
     'DEATH (MURDER-999.30)',
     'DEATH (TERM ILLNESS-999.40)',
     'DEATH (OTHER-999.50)',
     'DEATH (ACCIDENT-999.10)']
    tc = ['EVALUATION COMPLETED']
    df['treatment_complete'] = df['discharge_reason']
    df['dead'] = df['discharge_reason']
    df['treatment_complete'] = df['treatment_complete'].apply(mh_to_cat, args = (tc,))
    df['dead'] = df['dead'].apply(mh_to_cat, args = (death_list,))

def add_is_male_dummy(df):
    df['is_male'] = 0
    for i, row in df.iterrows():
        if row['sex'] == 'MALE':
            df.loc[i,'is_male'] = 1

def label_maker(df, label, delta = 1,):
    df[label] = 0
    for idx, row in df.iterrows():
        #print(row['rel_date'])
        try:
            r = datetime.strptime(row['rel_date'], '%Y-%m-%d')
            t = r + relativedelta(years=+delta)
            d_id = row['dedupe_id']
            for idx2, row2 in df.iterrows():
                if d_id == row2.dedupe_id and datetime.strptime(row2.bk_dt, '%Y-%m-%d') < t and datetime.strptime(row2.rel_date, '%Y-%m-%d') > r:
                    df.loc[idx, label] = 1
        except Exception as ex:
            pass

def label_maker2(df, label, delta = 1, picklefile = 'dfbL.p'):
    dfg = df.groupby('dedupe_id').count()
    dfg2 = dfg[dfg['bk_dt'] > 1]
    df['dup'] = df.dedupe_id.isin(dfg2.index)
    df_to_dic = df[df['dup']==True]
    d = {} # dict will have only instances with > 1 booking and
    # having a not NaN release date: much smaller set than the whole df
    for idx, row in df.iterrows():
        if idx not in d:
            try:
                r = row['rel_date']
                t = r + relativedelta(years=+delta)
                d[idx] = [row['dedupe_id'], r, t, row['bk_dt']]
            except Exception as ex:
                pass

    rep_set = set()
    for k1 in d:
        for k2 in d: # also can try with itertools.permutations
    '''
    dedup1 = d[k1][0]
    dedup2 = d[k2][0]
    releasedt1 = d[k1][1]
    releasedt2 = d[k2][1]
    bookdt2 = d[k2][3]
    t = d[k1][2]
    Basically test the same cond as before but in the dict
    '''
            if d[k1][0] == d[k2][0] and d[k2][3] > d[k1][2] and d[k2][1] > d[k1][1]:
                rep_set.add(k1) # add the index to the set

    rep_lst = list(rep_set)
    df[label] = df.index.isin(rep_lst) # is the index in the  list? -> label true
    df[label] = np.where(df[label]== True, 1, 0) # True = 1
    pickle.dump(df,open(picklefile, "wb"))

def binarize(df, var, var_lst, drop_orig=True):
    '''
    Construct binary valued-columns for cathegorical data.
    var_lst (lst) : values we care about, others omitted as if 'others'
    '''
    for val in var_lst:
        name_d = str(val) + '_dum'
        df[name_d] = np.where(df[var]== val, 1, 0)
    if drop_orig:
        df.drop([var], inplace=True, axis = 1)


def fancy_impute(df, method='mice'):
    if method =='knn':
        df = pd.DataFrame(data=fancyimpute.KNN(3).complete(df), columns=df.columns, index=df.index)
    else:
        df = pd.DataFrame(data=fancyimpute.MICE().complete(df), columns=df.columns, index=df.index)
    return df

def standarize(v,mn,st):
    return (v - mn) / st

def run_simple_loop(x_train, x_test, y_train, y_test):

    # define grid to use: test, small, large
    grid_size = 'test'
    clfs, grid = define_clfs_params(grid_size)

    # define models to run
    models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']

    # call clf_loop and store results in results_df
    results_df = clf_loop(models_to_run, clfs,grid, x_train, x_test, y_train, y_test)

    return results_df

def plot_importances(x_train, y_train, clf, n=10, LR_or_SVM=False, title='', get_coefs=False):
    '''
    Fit a classifier (the best) to the train set
    compute the relative importance of selected features in
    predicting the label.

    Inputs:
    - x_train, y_train
    - clf (Classifier())
    - features (lst of str)
    - label (str)
    - n (int): top n features, opt
    - title (str)
    '''
    clf.fit(x_train, y_train)
    if LR_or_SVM:
        importances = abs(clf.coef_[0])
    else:
        importances = clf.feature_importances_
    np_features = np.array(x_train.columns)
    sorted_idx = np.argsort(importances)[len(np_features)-n:]
    padding = np.arange(len(sorted_idx)) + 0.5
    plt.barh(padding, importances[sorted_idx], align='center')
    plt.yticks(padding, np_features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Relative Feature Importance (Top {})".format(n))
    plt.show()
    if get_coefs:
        return importances


def get_tops_to_dummy(train_df, cols_to_narrow, threshold, max_options = 5):
    values_to_keep = []
    counter = 1
    dummies_dict = {}

    for col in cols_to_narrow:
        col_sum = train_df[col].value_counts().sum()
        top = train_df[col].value_counts().nlargest(max_options)
        top_idx = list(top.index)
        top_value = 0
        num_dummies = int(0)

        while ((top_value / col_sum) < threshold) & (num_dummies < max_options):
            top_value += top[top_idx[num_dummies]]

            num_dummies += int(1)

        keep_dummies = top_idx[:num_dummies]
        dummies_dict[col] = keep_dummies

    counter += 1
    values_to_keep.append(dummies_dict)

    return values_to_keep



def apply_tops(values_to_keep, train_df, test_df = None):
    counter = 0
    for set_dict in values_to_keep:
        counter += 1
        for col, vals in set_dict.items():
            train_df.loc[~train_df[col].isin(vals), col] = 'Other'
            if test_df is not None:
                test_df.loc[~test_df[col].isin(vals), col] = 'Other'


def dummify_mh(df):
    ANXIETY_WORDS = ['ANXIETY', 'ANX']
    DEPRESSION_WORDS = ['DEPRESSIVE']
    PSYCOTIC_WORDS = ['BIPOLAR', 'SCHIZ']
    PTSD_WORDS = ['TRAUMATIC', 'PTSD']
    OPPS_WORDS = ['OPPOSITIONAL']
    DRUGS_WORDS = ['DRUG','SUBST', 'CANNA', 'OPOI', 'AMPH', 'STIM', 'COCAI', 'INHAL', 'ALCOHOL']

    df['anxiety_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (ANXIETY_WORDS,))

    df['depression_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (DEPRESSION_WORDS,))

    df['psychotic_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (PSYCOTIC_WORDS,))

    df['ptsd_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (PTSD_WORDS,))

    df['opps_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (OPPS_WORDS,))

    df['drugs_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (DRUGS_WORDS,))

def get_demog(df):
    #zips = list(df['zip'].unique())[1:]
    #educ_dic, pov_dic, unemp_dic = {}, {}, {}
    #for z in zips:
     #   educ_dic[z] = get_census.get_educzip(z)
     #   pov_dic[z] = get_census.get_povz(z)
      #  unemp_dic[z] = get_census.get_unempz(z)
    educ_dic = pickle.load(open("educ_dic.p", "rb"))
    pov_dic = pickle.load(open("pov_dic.p", "rb"))
    unemp_dic = pickle.load(open("unemp_dic.p", "rb"))
    return educ_dic, pov_dic, unemp_dic

def filter_df_by_date_range(df, date_var, start, end):
    df = df[(df[date_var] >= start) & (df[date_var] <= end)]
    return df

def temporal_train_test_data_split(df, features, filter_var, outcome_var,  start_date):
    if start_date == '2011-06-31':
        training_end_date = '2011-06-31'
        testing_end_date = '2012-06-31'

    elif start_date == '2012-01-01':
        training_end_date = '2012-01-01'
        testing_end_date = '2013-01'

    y_df = df[[filter_var, outcome_var]]

    x_train_filter_by_date = filter_df_by_date_range(df, filter_var, training_end_date, testing_end_date)
    x_test_filter_by_date = filter_df_by_date_range(df, filter_var, start_date, training_end_date)

    y_train_filter_by_date = filter_df_by_date_range(y_df, filter_var, training_end_date, testing_end_date)
    y_test_filter_by_date = filter_df_by_date_range(y_df, filter_var, start_date, training_end_date)

    x_train = x_train_filter_by_date[features]
    x_test = x_test_filter_by_date[features]

    y_train = y_train_filter_by_date[outcome_var]
    y_test = y_test_filter_by_date[outcome_var]

    return x_test, x_train, y_test, y_train


def find_best_model(results_df, criteria='p_at_1'):
    best_val = 0
    best_model = []
    best_params = []
    for idx, m in results_df.iterrows():
        if m[criteria] > best_val:
            best_val = m[criteria]
            best_model = m['clf']
            best_params = m['parameters']
            
    return best_model

def check_clf(clf):
    str_clf = str(clf)
    if str_clf[0] == 'L':
        return True
    return False

def get_top_risk(x_test, clf, thresh=0.5, topN =200):
    '''
    Returns the top N highest risk individuals, given 
    a preferred classifier (clf)
    - clf: instantiated classifier 
    e.g. clf = LogisticRegression(penalty='l1', C = param)
    '''
    y_scores =  clf.predict_proba(x_test)[:,1]
    dfinal = x_test
    dfinal['label_value'] = y_test
    dfinal['scores'] = y_scores
    dfinal['score'] = np.where(dfinal['scores'] > thresh, 1, 0)
    dfinal = dfinal.sort_values(by='scores', ascending=False)
    df_id = df1[['dedupe_id']]
    dF = pd.merge(df_id, dfinal, left_index=True, right_index=True)
    dF = dF.sort_values(by='scores', ascending=False)
    return dF.head(topN)

def process(df, vars_of_interest):
    # df['pri_dx_value'] = df['pri_dx_value'].str.upper()
    # df['anxiety_dummy'] = df['pri_dx_value']
    # df['depression_dummy'] = df['pri_dx_value']
    # df['psychotic_dummy'] = df['pri_dx_value']
    # df['ptsd_dummy'] = df['pri_dx_value']
    # df['opps_dummy'] = df['pri_dx_value']
    # df['drugs_dummy'] = df['pri_dx_value']

    # anxiety_words_list = ['ANXIETY', 'ANX']
    # df['anxiety_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (anxiety_words_list,))
    # depression_words_list = ['DEPRESSIVE']
    # df['depression_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (depression_words_list,))
    # psychotic_words_list = ['BIPOLAR', 'SCHIZ']
    # df['psychotic_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (psychotic_words_list,))
    # ptsd_words_list = ['TRAUMATIC', 'PTSD']
    # df['ptsd_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (psychotic_words_list,))
    # opps_words_list = ['OPPOSITIONAL']
    # df['opps_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (opps_words_list,))
    # drugs_words_list = ['DRUG','SUBST', 'CANNA', 'OPOI', 'AMPH', 'STIM', 'COCAI', 'INHAL', 'ALCOHOL']
    # df['drugs_dummy'] = df['pri_dx_value'].apply(mh_to_cat, args = (drugs_words_list,))

    dummify_mh(df)
    #add_is_male_dummy(df)

    binarize(df,'mar_stat', ['S', 'M', 'D', 'W'])
    binarize(df,'race',['WHITE', 'BLACK OR AFRICAN AMERICAN', 'ASIAN'])
    binarize(df,'sex',['MALE'])

    unemp_dict, educ_dict, pov_dict = get_demog(df)
    df['unemp'] = df['zip'].map(unemp_dict)
    df['educ'] = df['zip'].map(educ_dict)
    df['pov'] = df['zip'].map(pov_dict)

    zips = get_tops_to_dummy(df, cols_to_narrow = ['zip'], threshold = .8, max_options = 5)
    apply_tops(zips, mh_bookings)
    df = pd.get_dummies(df, columns = ['zip'], prefix= 'dum_', dummy_na = True)
    # after we merge queries, replace with the below:
    # df = pd.get_dummies(df, columns = ['zip', 'crime_class'], dummy_na = True)

    vars_to_norm = ['age_calced', 'unemp', 'educ', 'pov']
    for var in vars_to_norm:
        df[var] = df[var].apply(standarize, args=(df[var].mean(), df[var].std()))

    mlf.convert_true_false_1_0(df)

    label_maker2(df, 'returner')

    r_df = df[vars_of_interest] # features we will want to use (i.e.
    # just the numerical and the dummies) + label
    r_df = fancy_impute(r_df)


    x_train, x_test, y_train, y_test  = temporal_train_test_data_split(df, r_df, 'bk_dt	', 'returner',  '2011-06-31')
    results_df = run_simple_loop(x_train, x_test, y_train, y_test)
    best_clf = find_best_model(results_df, criteria='p_at_1')
    is_LR = check_clf(best_clf)
    
    plot_importances(x_train, y_train, best_clf, n=10, LR_or_SVM=is_LR, title='', get_coefs=False)
    
    top_risk = get_top_risk(x_test, best_clf, thresh=0.5, topN =200)

if __name__ == "__main__":
    mental_health_query += ';'
    mh_df = run_query(mental_health_query, conn)
    process(mh_df)
