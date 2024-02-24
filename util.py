import time
import dgl
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import os
import random


def read_list_graph_ho(num, file='drug_drug.csv', threshold=0.5, ablation=False):
    """
    read a list from a file as a graph
    :param num: the node number
    :param file: the list file path
    :param threshold: remove links with weights below this threshold
    :param ablation: use or unuse the information
    :return: a graph
    """
    start_time = time.time()
    data_dd = pd.read_csv(file, header=0)
    if ablation:
        dd_list = np.array([[i, i, 1] for i in range(num)])
    else:
        dd_list = np.array(data_dd.values.tolist() + [[i, i, 1] for i in range(num)])
    dd_list = dd_list[dd_list[:, 2] > threshold]

    ds = dd_list[:, 0].astype(int)
    do = dd_list[:, 1].astype(int)
    dw = dd_list[:, 2]
    dd_mat = np.zeros((num, num))
    for i in range(len(dd_list)):
        if dd_list[i, 2] > threshold:
            dd_mat[ds[i], do[i]] = dd_list[i, 2]

    graph_dr = dgl.graph((torch.tensor(ds), torch.tensor(do)))
    graph_dr.edata['weight'] = torch.tensor(dw, dtype=torch.float32)
    end_time = time.time()
    print(end_time - start_time)
    return graph_dr, dd_mat  # undirected graph


def get_data_idxs(data, entity_idxs1, entity_idxs2):
    """get indexes of all edges in a data set"""
    data_idxs = [(entity_idxs1[data[i][0]], entity_idxs2[data[i][1]]) for i in range(len(data))]
    return data_idxs


# def read_list_graph_he(file='data_indication.csv', type1='x_id', type2='y_name', neg_ratio=1):
#     """
#     read a list from a file as a graph with two type of nodes
#     :param file: the file path including a list of associations between drug and disease
#     :param type1: specific the first column name
#     :param type2: specific the second column name
#     :param neg_ratio:
#     :return:
#     """
#     start_time = time.time()
#     data_dd = pd.read_csv(file, header=0, encoding='unicode_escape', sep='\t')
#     drugs = list(set(data_dd[type1]))
#     diseases = list(set(data_dd[type2]))
#
#     entity_idxs1 = {drugs[i]: i for i in range(len(drugs))}
#     entity_idxs2 = {diseases[i]: i for i in range(len(diseases))}
#
#     dd_list = get_data_idxs(data_dd.values, entity_idxs1, entity_idxs2)
#     dd_list = list(set(dd_list))
#     dd_list = np.array(dd_list, dtype=np.int64)
#     dr_num = len(drugs)
#
#     dd_list[:, 1] += dr_num
#     ds = dd_list[:, 0].astype(int)
#     do = dd_list[:, 1].astype(int)
#
#     edges_re = dd_list
#     graph_dr = dgl.graph((torch.tensor(ds, dtype=torch.int64), torch.tensor(do, dtype=torch.int64)))
#
#     neg_edges = np.tile(dd_list, (neg_ratio, 1))
#     values1 = np.random.randint(len(drugs), size=dd_list.shape[0] * neg_ratio)
#     values2 = np.random.randint(len(diseases), size=dd_list.shape[0] * neg_ratio) + dr_num
#     choices = np.random.uniform(size=dd_list.shape[0] * neg_ratio)
#     subj = choices > 0.5
#     obj = choices <= 0.5
#     neg_edges[subj, 0] = values1[subj].tolist()
#     neg_edges[obj, 1] = values2[obj].tolist()
#     end_time = time.time()
#     print(end_time - start_time)
#     return graph_dr, edges_re, neg_edges, len(drugs), len(diseases)  # directed graph


def read_mat_graph_he(file='drug_disease.csv', neg_ratio=2):
    """
    read a matrix from a file as a graph
    :param file: specific the file path
    :param neg_ratio: randomly generate negative edges, which is equal to neg_ratio*positive edges
    :return: a graph and a list of edges
    """
    # import drug-disease association
    drug_dis_matrix = pd.read_csv(file, header=None, index_col=None)

    index_matrix = np.array(np.where(drug_dis_matrix == 1)).T
    no_index_matrix = np.array(np.where(drug_dis_matrix == 0)).T

    dr_num = drug_dis_matrix.shape[0]
    ds_num = drug_dis_matrix.shape[1]
    index_matrix[:, 1] += dr_num
    no_index_matrix[:, 1] += dr_num
    edges_re = index_matrix
    graph_h = dgl.graph((torch.tensor(index_matrix[:, 0]), torch.tensor(index_matrix[:, 1])))
    neg_edges = np.tile(index_matrix, (neg_ratio, 1))
    values1 = np.random.randint(drug_dis_matrix.shape[0], size=index_matrix.shape[0] * int(neg_ratio / 2))
    values2 = np.random.randint(drug_dis_matrix.shape[1], size=index_matrix.shape[0] * int(neg_ratio / 2)) + dr_num

    neg_edges[0:index_matrix.shape[0], 0] = values1.tolist()
    neg_edges[index_matrix.shape[0]: index_matrix.shape[0] * neg_ratio, 1] = values2.tolist()
    return graph_h, edges_re, neg_edges, no_index_matrix, dr_num, ds_num, np.array(drug_dis_matrix)


class EarlyStopping(object):
    """
    Early stopping to stop the training when the loss and auc does not improve.
    """

    def __init__(self, run_name, patience=30, saved_path='.'):
        self.filename = os.path.join(saved_path, 'early_stop_{}.pth'.format(run_name))
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        print(self.filename)
        model.load_state_dict(torch.load(self.filename))


def get_metrics_pre_sen_spe(y_valid, valid_pred_score):
    # calculate accuracy
    y_valid = y_valid.reshape(-1)
    valid_pred_score = valid_pred_score.reshape(-1)

    thresholds = []
    acc, sen, spe = [], [], []
    for i in range(10):
        rank = int(y_valid.sum()) * (i + 1)
        indices = np.argsort(valid_pred_score)[-rank:]
        pred_label = np.zeros_like(valid_pred_score)
        pred_label[indices[0:rank]] = 1
        # calculate accuracy
        acc.append(accuracy_score(y_valid, pred_label))
        # calculate sensitivity
        sen.append(recall_score(y_valid, pred_label))
        # calculate specificity
        spe.append(recall_score(y_valid, pred_label, pos_label=0))
    return acc, sen, spe


def get_metrics_global(real_score, predict_score):
    """
    calculate AUC and AUPR for ranking all potential disease-drug pairs
    :param real_score: the labels of drug-disease pairs
    :param predict_score: the predicted scores of drug-disease pairs
    :return: AUC and AUPR
    """
    AUC = roc_auc_score(real_score, predict_score)
    AUPR = average_precision_score(real_score, predict_score)
    return [AUC, AUPR]


def get_metrics_local(edges_ori, target, pred, dr_num, ds_num):
    """
    calculate AUC, AUPR, Hits@k, acc, sen, spe for ranking potential indications for each drug
    :param edges_ori: all edges
    :param target: the true labels
    :param pred: the predicted scores
    :param dr_num: number of drugs
    :param ds_num: number of diseases
    :return: UC, AUPR, Hits@k, acc, sen, spe
    """
    pred_mat = np.zeros((dr_num, ds_num))
    label_mat = np.zeros((dr_num, ds_num))
    for i in range(edges_ori.shape[0]):
        pred_mat[edges_ori[i, 0], edges_ori[i, 1]] = pred[i]
        label_mat[edges_ori[i, 0], edges_ori[i, 1]] = target[i]

    sort_values, sort_idxs = torch.sort(torch.tensor(pred_mat), dim=1, descending=True)
    sort_idxs = sort_idxs.cpu().numpy()

    def _get_auc():  # calculate AUC
        auc = []
        for i in range(dr_num):
            if np.sum(label_mat[i, :]) > 0:
                auc.append(roc_auc_score(label_mat[i, :], pred_mat[i, :]))
        return np.mean(auc)

    def _get_aupr():  # calculate AUPR
        aupr = []
        for i in range(dr_num):
            if np.sum(label_mat[i, :]) > 0:
                aupr.append(average_precision_score(label_mat[i, :], pred_mat[i, :]))
        return np.mean(aupr)

    def _get_hits():  # calculate Hits@k
        hits = []
        level = 0
        for i in range(dr_num):
            sign = 0
            if np.sum(label_mat[i, :]) > 0:
                hits.append([])
                for j in range(ds_num):
                    if label_mat[i, sort_idxs[i, j]] == 1:
                        sign = 1
                    hits[level].append(sign)
                level += 1
        hits = np.array(hits)
        return np.mean(hits[:, 0]), np.mean(hits[:, 2]), np.mean(hits[:, 9])

    def _get_acc():  # calculate acc, spe, sen
        acc_all = []
        sen_all = []
        spe_all = []
        for i in range(dr_num):
            if np.sum(label_mat[i, :]) > 0:
                thresholds = []
                for j in range(10):
                    # calculate the threshold
                    rank = (j + 1)
                    thresholds.append(np.partition(pred_mat[i, :], -rank)[-rank])

                acc, sen, spe = [], [], []
                for threshold in thresholds:
                    pred_label = np.where(pred_mat[i, :] >= threshold, 1, 0)
                    acc.append(accuracy_score(label_mat[i, :], pred_label))
                    # calculate sensitivity
                    sen.append(recall_score(label_mat[i, :], pred_label))
                    # calculate specificity
                    spe.append(recall_score(label_mat[i, :], pred_label, pos_label=0))
                acc_all.append(acc)
                sen_all.append(sen)
                spe_all.append(spe)

        return np.mean(np.array(acc_all), axis=0), np.mean(np.array(sen_all), axis=0), np.mean(np.array(spe_all),
                                                                                               axis=0)

    auc = _get_auc()
    aupr = _get_aupr()
    hits1, hits3, hits10 = _get_hits()
    acc, sen, spe = _get_acc()

    return [hits1, hits3, hits10, auc, aupr] + acc.tolist() + sen.tolist() + spe.tolist()


def split_data_hov(edge_index, neg_edge_index, split=[8, 8, 9]):
    """
    split data into train, valid, test
    :param edge_index: all positive samples
    :param neg_edge_index: all negative samples
    :param split: split[0]-split[1]: train, split[1]-split[2]: valid, split[2]-end: test
    :return: training, validation, and testing data
    """
    idxs = [i for i in range(edge_index.shape[0])]
    random.shuffle(idxs)
    train_idxs_pos = idxs[:(len(idxs) // 10) * split[0]]
    valid_idxs_pos = idxs[(len(idxs) // 10) * split[1]: (len(idxs) // 10) * split[2]]
    test_idxs_pos = idxs[(len(idxs) // 10) * split[2]:]
    idxs_neg = [i for i in range(neg_edge_index.shape[0])]
    random.shuffle(idxs_neg)
    train_idxs_neg = idxs_neg[:(len(idxs_neg) // 10) * split[0]]
    valid_idxs_neg = idxs_neg[(len(idxs_neg) // 10) * split[1]: (len(idxs_neg) // 10) * split[2]]
    test_idxs_neg = idxs_neg[(len(idxs_neg) // 10) * split[2]:]
    return [train_idxs_pos, valid_idxs_pos, test_idxs_pos], [train_idxs_neg, valid_idxs_neg, test_idxs_neg]


def save_results(metric, valid_metric, file_out):
    """
    save results of vlaidation and testing data in a text file
    :param metric: results on test data
    :param valid_metric: results on validation data
    :param file_out: the path of the text file
    """
    with open(file_out, 'w') as file:
        file.write("validation")
        for i in range(len(valid_metric[0]) - 1):
            file.write(",")
        file.write('\n')
        for i in valid_metric:
            for j in range(len(i) - 1):
                file.write(str(i[j]) + ',')
            file.write(str(i[-1]))
            file.write('\n')
        valid_mean = np.mean(valid_metric, axis=0)
        for j in range(len(valid_mean) - 1):
            file.write(str(valid_mean[j]) + ',')
        file.write(str(valid_mean[-1]))
        file.write('\n')

        file.write("test")
        for i in range(len(metric[0]) - 1):
            file.write(",")
        file.write('\n')
        for i in metric:
            for j in range(len(i) - 1):
                file.write(str(i[j]) + ',')
            file.write(str(i[-1]))
            file.write('\n')
        test_mean = np.mean(metric, axis=0)
        for j in range(len(test_mean) - 1):
            file.write(str(test_mean[j]) + ',')
        file.write(str(test_mean[-1]))
