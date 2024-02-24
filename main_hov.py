from torch_optimizer import RAdam
from model import Model
from util import *
import numpy as np
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='DrIGCL')
parser.add_argument('--bert_n_heads', type=int, default=4,
                    help='number of heads in self-attention')
parser.add_argument('--n_layers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--hidden1', type=int, default=256,
                    help='dimension of hidden layer 1')
parser.add_argument('--drop_out_rating', type=float, default=0.2,
                    help='drop out rate')
parser.add_argument('--loss_weight', type=float, default=0.4,
                    help='weight of contrastive loss')
parser.add_argument('--cl_weight', type=float, default=1,
                    help='weight of classification loss')
parser.add_argument('--mp', type=float, default=0.9,
                    help='margin in contrastive loss')
parser.add_argument('--split_ratio', type=list, default=[8, 8, 9],
                    help='split ratio of training, validation, and testing')
parser.add_argument('--epo_num', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--runs', type=int, default=20,
                    help='number of runs')
parser.add_argument('--learn_rating', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--file_path', type=str, default='..\\results\\res',
                    help='path to save models')
parser.add_argument('--patience', type=int, default=30,
                    help='patience of early stopping')
parser.add_argument('--batch_s', type=int, default=1000,
                    help='batch size')
parser.add_argument('--drug_ablation', type=bool, default=False,
                    help='drug ablation')
parser.add_argument('--disease_ablation', type=bool, default=False,
                    help='disease ablation')

args = parser.parse_args()


def experiments():
    """
    The experiments require hyperparameters from args
    The experiments require the following files: drug_disease.csv, drug_drug.csv, dis_dis_rem.csv.
    The experiments save metrics including AUC, Hits@1, Hits@3, Hits@10, accuracy, recall, and specificity.
    """
    hidden2 = int(args.hidden1 / 2)
    graph_h, edges_re, neg_edges, no_index_matrix, dr_num, ds_num, drug_dis_matrix = read_mat_graph_he(
        '..\\results\\data\\drug_disease.csv')
    graph_dr, dr_mat = read_list_graph_ho(dr_num, '..\\results\\data\\drug_drug.csv', threshold=0.,
                                          ablation=args.drug_ablation)
    graph_ds, ds_mat = read_list_graph_ho(ds_num, '..\\results\\data\\dis_dis_rem.csv', threshold=0.5,
                                          ablation=args.disease_ablation)
    dr_ds_num = dr_num + ds_num
    event_num = 2

    dr_mat = torch.tensor(dr_mat, dtype=torch.float32).to(device)
    ds_mat = torch.tensor(ds_mat, dtype=torch.float32).to(device)

    edge_index = edges_re.copy()
    neg_edge_index = neg_edges.copy()

    metric = []
    valid_metric = []
    fold = 0

    for i in range(args.runs):
        idxs_pos, idxs_neg = split_data_hov(edge_index, neg_edge_index, split=args.split_ratio)

        train_pos_index = idxs_pos[0]
        valid_pos_index = idxs_pos[1]
        test_pos_index = idxs_pos[2]
        train_neg_idx = idxs_neg[0]
        valid_neg_index = idxs_neg[1]
        test_neg_idx = idxs_neg[2]

        train_edge_index, valid_edge_index, test_edge_index = edge_index[train_pos_index], edge_index[valid_pos_index], edge_index[test_pos_index]
        neg_train_edge_index, neg_valid_edge_index, neg_test_edge_index = neg_edges[train_neg_idx], neg_edges[valid_neg_index], neg_edges[test_neg_idx]
        train_edges = np.vstack([train_edge_index, neg_train_edge_index])
        valid_edges = np.vstack([valid_edge_index, no_index_matrix])
        test_edges = np.vstack([test_edge_index, no_index_matrix])
        # reassign index of drug and disease after splitting train validation and test sets
        train_edges_ori = train_edges.copy()
        train_edges_ori[:, 1] -= dr_num
        valid_edges_ori = valid_edges.copy()
        valid_edges_ori[:, 1] -= dr_num
        test_edges_ori = test_edges.copy()
        test_edges_ori[:, 1] -= dr_num

        y_train = np.concatenate([np.ones(len(train_edge_index)), np.zeros(len(neg_train_edge_index))])
        y_valid = np.concatenate([np.ones(len(valid_edge_index)), np.zeros(no_index_matrix.shape[0])])
        y_test = np.concatenate([np.ones(len(test_edge_index)), np.zeros(no_index_matrix.shape[0])])
        target = torch.tensor(y_train, dtype=torch.float32).to(device)

        train_edge_idx = torch.tensor(train_pos_index, dtype=torch.int64)
        train_graph = dgl.edge_subgraph(graph_h, train_edge_idx, relabel_nodes=False)
        train_graph = dgl.to_bidirected(train_graph)

        graph_dr = graph_dr.to(device)
        graph_ds = graph_ds.to(device)
        train_graph = train_graph.to(device)

        # train_graph
        edges_ori = torch.tensor(train_edges_ori).to(device)
        valid_edges_ori = torch.tensor(valid_edges_ori).to(device)
        t_edges_ori = torch.tensor(test_edges_ori).to(device)

        model = Model(dr_num, ds_num, dr_ds_num, args.bert_n_heads, args.n_layers, args.hidden1, hidden2,
                      args.drop_out_rating, args.mp)

        len_train = len(y_train)
        len_test = len(y_test)
        print("arg train len", len_train)
        print("test len", len_test)
        model_optimizer = RAdam(model.parameters(), lr=args.learn_rating, weight_decay=0.0001)
        model = model.to(device)

        stopper = EarlyStopping(i, patience=args.patience, saved_path=args.file_path)

        for epoch in range(args.epo_num):
            model.train()
            # X:predict model value;
            X, CFV, rec_dr_sim, rec_ds_sim, rec_drds = model(graph_dr, graph_ds, train_graph, edges_ori)
            pred = torch.sigmoid(X)
            #rec_he = index_select(rec_drds, edges_ori).to(device)
            loss = model.get_loss(X, target, CFV, args.loss_weight, args.cl_weight)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            model.eval()
            AUC_, _ = get_metrics_global(target.cpu().detach().numpy(),
                                         pred.cpu().detach().numpy())
            print('Epoch {} Loss: {:.5f}; Train AUC {:.5f}'.
                  format(epoch, loss.item(), AUC_))

            early_stop = stopper.step(loss.item(), AUC_, model)
            if early_stop:
                print("Early stopping!")
                break

        stopper.load_checkpoint(model)

        model.eval()
        with torch.no_grad():
            valid_pred_score = np.empty(shape=(0, 1))
            for i in range(valid_edges_ori.shape[0] // args.batch_s + 1):
                valid_temp = valid_edges_ori[i * args.batch_s: (i + 1) * args.batch_s]
                valid_X, _, _, _, _ = model(graph_dr, graph_ds, train_graph, valid_temp)
                valid_pred_score_temp = torch.sigmoid(valid_X).cpu().detach().numpy()
                valid_pred_score = np.vstack([valid_pred_score, valid_pred_score_temp])

            valid_result_all1, _ = get_metrics_global(y_valid, valid_pred_score)
            valid_local_metric = get_metrics_local(valid_edges_ori.cpu().detach().numpy(), y_valid, valid_pred_score,
                                                   dr_num, ds_num)

            print('Fold {} ; Valid AUC {:.5f}; Hits@1 {:.5f}; Hits@3 {:.5f}; Hits@10 {:.5f}'.
                  format(fold, valid_result_all1, valid_local_metric[0], valid_local_metric[1], valid_local_metric[2]))
            valid_metric.append([valid_result_all1] + valid_local_metric)

            pred_score = np.empty(shape=(0, 1))
            for i in range(t_edges_ori.shape[0] // args.batch_s + 1):
                temp = t_edges_ori[i * args.batch_s:(i + 1) * args.batch_s]
                X_t, _, _, _, rec_drds_t = model(graph_dr, graph_ds, train_graph, temp)
                pred_score_temp = torch.sigmoid(X_t).cpu().detach().numpy()
                pred_score = np.vstack([pred_score, pred_score_temp])

            result_all1, _ = get_metrics_global(y_test, pred_score)
            local_metric = get_metrics_local(t_edges_ori.cpu().detach().numpy(), y_test, pred_score, dr_num, ds_num)

            print('Fold {} ; Test AUC {:.5f}; Hits@1 {:.5f}; Hits@3 {:.5f}; Hits@10 {:.5f}'.format(fold, result_all1,
                                                                                                   local_metric[0],
                                                                                                   local_metric[1],
                                                                                                   local_metric[2]))
            metric.append([result_all1] + local_metric)
        fold += 1

    print(metric)

    file_out = (
        "..\\results\\results_revision\\ratio_{}_DrIGCL_layers_{}_dropout_{}_embedding_{}_epochs{}_lw{}_cl{}_mp{}.txt".
        format(args.split_ratio[0], args.n_layers, args.drop_out_rating, args.hidden1, args.epo_num, args.loss_weight,
               args.cl_weight, args.mp))

    save_results(metric, valid_metric, file_out)


if __name__ == "__main__":
    experiments()
