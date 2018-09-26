from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def joint_sort_descending(l1, l2):
    '''
    Attribution: Adapted from code by Rayid Ghani, https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    # l1 and l2 have to be numpy arrays
#     if not isinstance(l1, (np.ndarray)):
#         l1 = np.array(l1)
#     if not isinstance(l2, (np.ndarray)):
#         l1 = np.array(l2)

    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]



def generate_binary_at_k(test_pred, k):
    '''
    Attribution: Rayid Ghani, https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    cutoff_index = int(len(test_pred) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(test_pred))]
    return predictions_binary




def precision_at_k(testing_outcome, test_pred, k):
    '''
    Attribution: Rayid Ghani, 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(test_pred), np.array(testing_outcome))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision



def f1_at_k(testing_outcome, test_pred, k):
    '''
    Attribution: Adapted from prediction and recoll score calculation by Rayid Ghani,
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(test_pred), np.array(testing_outcome))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    f1 = f1_score(y_true_sorted, preds_at_k)
    return f1



def scores_at_k(y_true, y_scores, k):
    '''
    Calculate precision, recall, and f1 score at a given threshold
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    return precision, recall, f1


def recall_at_k(testing_outcome, test_pred, k):
    '''
    Attribution: Rayid Ghani, 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(test_pred), np.array(testing_outcome))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


def plot_precision_recall_n(testing_outcome, test_pred, model_name):
    '''
    Attribution: Rayid Ghani, 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(testing_outcome, test_pred)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(test_pred)
    for value in pr_thresholds:
        num_above_thresh = len(test_pred[test_pred>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('Percentage of Population')
    ax1.set_ylabel('Precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('Recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()