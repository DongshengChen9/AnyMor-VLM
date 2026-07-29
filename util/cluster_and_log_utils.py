import torch
import torch.distributed as dist
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score



def all_sum_item(item):
    item = torch.tensor(item).cuda()
    dist.all_reduce(item)
    return item.item()
    
def hungarian_on_subset(w, gt_classes):
    if len(gt_classes) == 0:
        return None, 0, 0

    w_sub = w[:, gt_classes]

    row_ind, col_ind = linear_assignment(w_sub.max() - w_sub)

    correct = w_sub[row_ind, col_ind].sum()
    total = w_sub.sum()

    if total == 0:
        return None, np.nan, 0

    acc = correct / total
    return (row_ind, col_ind), acc, total


def split_cluster_acc_v4(y_true, y_pred, mask_seen, mask_new):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    assert y_pred.size == y_true.size

    mask_semi = (~mask_seen) & (~mask_new)

    old_classes_gt = set(y_true[mask_seen])
    semi_classes_gt = set(y_true[mask_semi])
    new_classes_gt = set(y_true[mask_new])

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    
    # --- Total ACC ---
    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = y_pred.size
    try: 
        if dist.get_world_size() > 0:
            total_acc = all_sum_item(total_acc)
            total_instances = all_sum_item(total_instances)
    except: pass
    total_acc /= total_instances

    # --- Old/Seen ACC ---
    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    try:
        if dist.get_world_size() > 0:
            old_acc = all_sum_item(old_acc)
            total_old_instances = all_sum_item(total_old_instances)
    except: pass
    old_acc = old_acc / total_old_instances if total_old_instances != 0 else np.nan

    # --- Semi ACC ---
    semi_acc = 0
    total_semi_instances = 0
    for i in semi_classes_gt:
        semi_acc += w[ind_map[i], i]
        total_semi_instances += sum(w[:, i])
    try:
        if dist.get_world_size() > 0:
            semi_acc = all_sum_item(semi_acc)
            total_semi_instances = all_sum_item(total_semi_instances)
    except: pass
    semi_acc = semi_acc / total_semi_instances if total_semi_instances != 0 else np.nan

    # --- New ACC ---
    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    try:
        if dist.get_world_size() > 0:
            new_acc = all_sum_item(new_acc)
            total_new_instances = all_sum_item(total_new_instances)
    except: pass
    new_acc = new_acc / total_new_instances if total_new_instances != 0 else np.nan


    # --- ari & nmi ---
    total_ari = adjusted_rand_score(y_true, y_pred)
    total_nmi = normalized_mutual_info_score(y_true, y_pred)

    if np.sum(mask_seen) > 0:
        old_ari = adjusted_rand_score(y_true[mask_seen], y_pred[mask_seen])
        old_nmi = normalized_mutual_info_score(y_true[mask_seen], y_pred[mask_seen])
    else:
        old_ari, old_nmi = np.nan, np.nan

    if np.sum(mask_semi) > 0:
        semi_ari = adjusted_rand_score(y_true[mask_semi], y_pred[mask_semi])
        semi_nmi = normalized_mutual_info_score(y_true[mask_semi], y_pred[mask_semi])
    else:
        semi_ari, semi_nmi = np.nan, np.nan

    if np.sum(mask_new) > 0:
        new_ari = adjusted_rand_score(y_true[mask_new], y_pred[mask_new])
        new_nmi = normalized_mutual_info_score(y_true[mask_new], y_pred[mask_new])
    else:
        new_ari, new_nmi = np.nan, np.nan

    results = {
        "acc":  {"total": total_acc, "old": old_acc, "semi": semi_acc, "new": new_acc},
        "ari":  {"total": total_ari, "old": old_ari, "semi": semi_ari, "new": new_ari},
        "nmi":  {"total": total_nmi, "old": old_nmi, "semi": semi_nmi, "new": new_nmi}
    }

    return results
    

EVAL_FUNCS = {
    'v4': split_cluster_acc_v4
}

def log_accs_from_preds_v4(y_true, y_pred, mask_seen, mask_new, eval_funcs, save_name, T=None,
                           print_output=False, args=None):
    """
    Given a list of evaluation functions to use, evaluate and log ACC, ARI, and NMI results.

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask_seen: Which instances belong to Old classes
    :param mask_new: Which instances belong to New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use (e.g. ['v3'])
    :param save_name: What are we evaluating on
    :param print_output: Whether to log/print the results
    :param args: Configuration arguments containing the logger
    :return: (all_acc, old_acc, semi_acc, new_acc) from the first evaluation function
    """

    mask_seen = mask_seen.astype(bool)
    mask_new = mask_new.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    to_return = None

    for i, f_name in enumerate(eval_funcs):
        acc_f = EVAL_FUNCS[f_name]
        
        res_dict = acc_f(y_true, y_pred, mask_seen, mask_new)
        log_name = f'{save_name}_{f_name}'

        if i == 0:
            to_return = (
                res_dict["acc"]["total"], 
                res_dict["acc"]["old"], 
                res_dict["acc"]["semi"], 
                res_dict["acc"]["new"],
                res_dict["ari"]["total"], 
                res_dict["ari"]["old"], 
                res_dict["ari"]["semi"], 
                res_dict["ari"]["new"],
                res_dict["nmi"]["total"], 
                res_dict["nmi"]["old"], 
                res_dict["nmi"]["semi"], 
                res_dict["nmi"]["new"]
            )

        if print_output:

            lines = [
                f'Epoch {T}, {log_name} [ACC]: All {res_dict["acc"]["total"]:.4f} | Old {res_dict["acc"]["old"]:.4f} | Semi {res_dict["acc"]["semi"]:.4f} | New {res_dict["acc"]["new"]:.4f}',
                f'Epoch {T}, {log_name} [ARI]: All {res_dict["ari"]["total"]:.4f} | Old {res_dict["ari"]["old"]:.4f} | Semi {res_dict["ari"]["semi"]:.4f} | New {res_dict["ari"]["new"]:.4f}',
                f'Epoch {T}, {log_name} [NMI]: All {res_dict["nmi"]["total"]:.4f} | Old {res_dict["nmi"]["old"]:.4f} | Semi {res_dict["nmi"]["semi"]:.4f} | New {res_dict["nmi"]["new"]:.4f}'
            ]
            

            print_str = "\n".join(lines)

            try:
                if dist.get_rank() == 0:
                    try:
                        args.logger.info(print_str)
                    except:
                        print(print_str)
            except:

                if print_output:
                    print(print_str)

    return to_return