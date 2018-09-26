import pandas as pd
import numpy as np
import itertools
from itertools import chain
import sklearn
from sklearn import preprocessing, svm, metrics, tree, decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import graphviz
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, f1_score, roc_auc_score
import magiclooping as mp



def split_data(df, outcome_var, geo_columns, test_size, seed = None):
    '''
    Separate data frame into training and test subsets based on specified size
    for model training and evaluation.

    Inputs:
        df: pandas dataframe
        outcome_var: (string) variable model will predict
        geo_columns:  (list of strings) list of column names corresponding to
            columns with numeric geographical information (ex: zipcodes)
        test_size: (float) proportion of data to hold back from training for
            testing

    Output: testing and training data sets for predictors and outcome variable
    '''
    # remove outcome variable and highly correlated variables
    all_drops = [outcome_var] + geo_columns
    X = df.drop(all_drops, axis=1)
    # isolate outcome variable in separate data frame
    Y = df[outcome_var]

    return train_test_split(X, Y, test_size = test_size, random_state = seed)


def temporal_train_test_split(df, outcome_var, exclude = [], keep_cols = False):
    if not keep_cols:
        skips = [outcome_var] + exclude
        Xs = df.drop(skips, axis = 1)
    else:
        Xs = df[keep_cols]

    Ys = df[outcome_var]

    return Xs, Ys


def develop_args(name, params_dict):
        # create dictionaries for each possible tuning option specified
        # in param_dict

    print("Creating args for: {} models".format(name))

    options = params_dict[name]
    tuners = list(options.keys())
    list_params = list(itertools.product(*options.values()))

    all_model_params = []

    for params in list_params:
        kwargs_dict = dict(zip(tuners, params))
        all_model_params.append(kwargs_dict)

    return all_model_params



def cf_loop(pred_train, label_train, pred_test, label_test, set_num, 
    thresholds = [5, 10, 20], params_dict = None, plot = False, which_clfs = None):
    '''
    Attribution: Adapted from Rayid Ghani's magicloop and simpleloop examples
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    result_cols = ['set_num', 'model_type','clf', 'parameters', 
                    'baseline_precision','baseline_recall','auc-roc']

    # define columns for metrics at each threshold specified in function call
    result_cols += list(chain.from_iterable(('p_at_{}'.format(threshold), 
        'r_at_{}'.format(threshold), 
        'f1_at_{}'.format(threshold)) for threshold in thresholds))

    # define dataframe to write results to
    results_df =  pd.DataFrame(columns=result_cols)

    all_clfs = {
        'DecisionTree': DecisionTreeClassifier(random_state=1008),
        'LogisticRegression': LogisticRegression(penalty='l1', C=1e5),
        'Bagging': BaggingClassifier(base_estimator=LogisticRegression(
            penalty='l1', C=1e5)),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=1008),
        'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
            algorithm="SAMME", n_estimators=200),
        'GradientBoosting': GradientBoostingClassifier(learning_rate=0.05, 
            subsample=0.5, max_depth=6, n_estimators=10),
        'NaiveBayes': GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    }

    if which_clfs:
        clfs = {clf: all_clfs.get(clf, None) for clf in which_clfs}
    else:
        clfs = all_clfs

    if params_dict is None:
        # define parameters to loop over. Thanks to the DSSG team for the 
        #recommendations!
        params_dict = {
            "DecisionTree": {'criterion': ['gini', 'entropy'], 
                'max_depth': [1,5,10,20,50,100], 
                'max_features': [None, 'sqrt','log2'],
                'min_samples_split': [2,5,10], 'random_state':[1008]},
            "LogisticRegression": { 'penalty': ['l1'], 'C': [0.01]},
            "Bagging": {},
            "SVM": {'C' :[0.01],'kernel':['linear']},
            "AdaBoost": { 'algorithm': ['SAMME'], 'n_estimators': [1]},
            'GradientBoosting': {'n_estimators': [1], 
                'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
            'NaiveBayes' : {},
            "RandomForest": {'n_estimators': [100, 10000], 
                'max_depth': [5,50], 'max_features': ['sqrt','log2'],
                'min_samples_split': [2,10], 'n_jobs':[-1], 
                'random_state':[1008]},
            "KNN": {'n_neighbors': [5],'weights': ['uniform'],
                'algorithm': ['auto']}
        }

    for name, clf in clfs.items():
        print("Creating classifier: {}".format(name))

        if clf is None:
            continue

        # create all possible models using tuners in dictionaries created above
        all_model_params = develop_args(name, params_dict)

        for args in all_model_params:
            try:
                clf.set_params(**args)
                y_pred_probs = clf.fit(pred_train, 
                    label_train.values.ravel()).predict_proba(pred_test)[:,1]

                y_pred_probs_sorted, y_test_sorted = mp.joint_sort_descending(
                    np.array(y_pred_probs), np.array(label_test))

                # print("Evaluating {} models".format(name))

                results_list = [set_num, name, clf, args, 
                    mp.precision_at_k(y_test_sorted, y_pred_probs_sorted, 100.0),
                    mp.recall_at_k(y_test_sorted, y_pred_probs_sorted, 100.0), 
                    roc_auc_score(label_test, y_pred_probs)]

                for threshold in thresholds:
                    precision, recall, f1 = mp.scores_at_k(y_test_sorted, 
                        y_pred_probs_sorted, threshold)
                    results_list += [precision, recall, f1]

                results_df.loc[len(results_df)] = results_list

                if plot:
                    mp.plot_precision_recall_n(label_test,y_pred_probs, clf)

            except Exception as e:
                print("Error {} on model {} with parameters {}".format(e, name, 
                                                                        args))
                print()
                continue

    return results_df


def temporal_train_test_split(df, outcome_var, exclude = [], subset_cols = False):
    if not subset_cols:
        skips = [outcome_var] + exclude
        Xs = df.drop(skips, axis = 1)
    else:
        Xs = df[subset_cols]

    Ys = df[outcome_var]

    return Xs, Ys


def run_models(train_test_tuples, outcome_var, clfs, ks = [5, 10, 20]):
    all_results = []
    for i, (train, test) in enumerate(train_test_tuples):
        print("set", i)

        # x_train, y_train = temporal_train_test_split(train, outcome_var)
        # x_test, y_test = temporal_train_test_split(test, outcome_var)
        # results = cf_loop(x_train, y_train, x_test, y_test,
        #                      ks = ks,
        #                      set_num = i, params_dict = None,
        #                      which_clfs = clfs)
        # all_results.append(results)

    return pd.concat(all_results, ignore_index = True)


def construct_best(metrics_df):
    identifiers = ['clf', 'parameters', 'model_type', 'set_num']
    metric_cols = set(metrics_df.columns) - set(identifiers)

    best_df = pd.DataFrame(columns = ['metric','baseline_p', 'max_value',
                                     'model_num', 'model_type', 'clf', 
                                     'test_set'])

    for col in metric_cols:
        best = metrics_df[col].max()
        idx = metrics_df[col].idxmax()
        row = [col, metrics_df.loc[idx, 'baseline_precision'],
               best, idx, metrics_df.loc[idx, 'model_type'],
               metrics_df.loc[idx, 'clf'], metrics_df.loc[idx, 'set_num']]

        best_df.loc[len(best_df)] = row

    best_df.set_index('metric', inplace = True)
    return best_df



def loop_dt(param_dict, training_predictors, testing_predictors,
                training_outcome, testing_outcome):
    '''
    Loop over series of possible parameters for decision tree classifier to
    train and test models, storing accuracy scores in a data frame

    Inputs:
        param_dict: (dictionary) possible decision tree parameters
        training_predictors: data set of predictor variables for training
        testing_predictors: data set of predictor variables for testing
        training_outcome: outcome variable for training
        testing_outcome: outcome variable for testing

    Outputs:
        accuracy_df: (data frame) model parameters and accuracy scores for
            each iteration of the model

    Attribution: adapted combinations of parameters from Moinuddin Quadri's
    suggestion for looping: https://stackoverflow.com/questions/42627795/i-want-to-loop-through-all-possible-combinations-of-values-of-a-dictionary
    and method for faster population of a data frame row-by-row from ShikharDua:
    https://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe
    '''


    rows_list = []
    for clf_type, classifier in classifier_type.items():

        for params in list(itertools.product(*param_dict.values())):
            classifier(params)
            dec_tree.fit(training_predictors, training_outcome)


    rows_list = []
    for params in list(itertools.product(*param_dict.values())):
        dec_tree = DecisionTreeClassifier(criterion = params[0],
                                          max_depth = params[1],
                                          max_features = params[2],
                                          min_samples_split = params[3])
        dec_tree.fit(training_predictors, training_outcome)

        train_pred = dec_tree.predict(training_predictors)
        test_pred = dec_tree.predict(testing_predictors)

        # evaluate accuracy
        train_acc = accuracy(train_pred, training_outcome)
        test_acc = accuracy(test_pred, testing_outcome)

        acc_dict = {}
        (acc_dict['criterion'], acc_dict['max_depth'], acc_dict['max_features'], 
        acc_dict['min_samples_split']) = params
        acc_dict['train_acc'] = train_acc
        acc_dict['test_acc'] = test_acc

        rows_list.append(acc_dict)

    accuracy_df = pd.DataFrame(rows_list)

    return accuracy_df


def create_best_tree(accuracy_df, training_predictors, training_outcome):
    '''
    Create decision tree based on highest accuracy score in model testing, to
    view feature importance of each fitted feature

    Inputs:
        accuracy_df: (data frame) model parameters and accuracy scores for
            each iteration of the model
        training_predictors: data set of predictor variables for training
        training_outcome: outcome variable for training

    Outputs:
        best_tree: (classifier object) decision tree made with parameters used
            for highest-ranked model in terms of accuracy score during
            parameters loop
    '''
    accuracy_ranked = accuracy_df.sort_values('test_acc', ascending = False)
    dec_tree = DecisionTreeClassifier(
    criterion = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'criterion'],
    max_depth = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'max_depth'],
    max_features = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 
                                        'max_features'],
    min_samples_split = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 
                                            'min_samples_split'])

    dec_tree.fit(training_predictors, training_outcome)

    return dec_tree


def feature_importance_ranking(best_tree, training_predictors):
    '''
    View feature importance of each fitted feature

    Inputs:
        best_tree: (classifier object) decision tree made with parameters used
            for highest-ranked model in terms of accuracy score during
            parameters loop

    Outputs:
        features_df: (data frame) table of feature importance for each
        predictor variable
    '''
    features_df = pd.DataFrame(best_tree.feature_importances_,
                                training_predictors.columns).rename(
                                columns = {0: 'feature_importance'}, 
                                inplace = True)
    features_df.sort_values(by = 'feature_importance', ascending = False)
    return features_df


def visualize_best_tree(best_tree, training_predictors):
    '''
    Visualize decision tree object with GraphWiz
    '''
    viz = sklearn.tree.export_graphviz(best_tree,
                    feature_names = training_predictors.columns,
                    class_names=['Financially Stable', 'Financial Distress'],
                    rounded=False, filled=True)

    with open("tree.dot") as f:
        dot_graph = f.read()
        graph = graphviz.Source(dot_graph)

    return graph
