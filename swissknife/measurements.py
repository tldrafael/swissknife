import warnings
import numpy as np


def get_CM(pred, label, n_classes):
    # https://github.com/Qualcomm-AI-research/InverseForm/blob/be142136087579d5f7175cbf64c171fc52352fc7/utils/misc.py#L20
    # I ADAPTED THE CODE TO OBEY THE CORRECT RESHAPING ORDER ORDER='F' (IT PROBABLY HAD A BUG BEFORE)
    CM_cur = np.bincount(n_classes * label.flatten() + pred.flatten(), minlength=n_classes ** 2)
    return CM_cur.reshape(n_classes, n_classes, order='F').astype(int)


def get_CM_fromloader_cityscapes(dloader, model, n_classes, ix_nolabel=255):
    # https://github.com/Qualcomm-AI-research/InverseForm/blob/be142136087579d5f7175cbf64c171fc52352fc7/utils/misc.py#L20
    # stretch ground truth labels by num_classes
    # TP at 0 + 0, 1 + 1, 2 + 2 ...  # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    CM_abs = np.zeros((n_classes, n_classes), dtype=int)
    for inp_data, inp_label in dloader:
        test_preds = model(inp_data.cuda()).argmax(1, keepdim=True).cpu()
        for pr_i, y_i in zip(test_preds, inp_label):
            if ix_nolabel is None:
                CM_abs += get_CM(pr_i, y_i, n_classes)
            else:
                if (pr_i == ix_nolabel).sum().item() > 0:
                    warnings.warn('The model has also predicted ix_nolabel value, these pixels are also being ignored '
                                  ' - thus, this metric is not correct!')
                mask = (y_i != ix_nolabel) & (pr_i != ix_nolabel)
                CM_abs += get_CM(pr_i[mask], y_i[mask], n_classes)
    return CM_abs


def get_CM_fromloader_cityscape_frombatch(inp_data, inp_label, model, n_classes):
    CM_abs = np.zeros((n_classes, n_classes), dtype=int)
    test_preds = model.forward(inp_data.cuda()).argmax(1).cpu()
    for i in range(inp_data.shape[0]):
        for pr_i, y_i in zip(test_preds[i], inp_label[i]):
            CM_abs += get_CM(pr_i, y_i, n_classes)
    return CM_abs


def get_miou_fromloader(dloader, model, n_classes, return_all=False, fl_singleimage=False, ix_nolabel=255):
    if not fl_singleimage:
        CM_abs = get_CM_fromloader_cityscapes(dloader, model, n_classes, ix_nolabel)
    else:
        CM_abs = get_CM_fromloader_cityscape_frombatch(*dloader, model, n_classes)

    pred_P = CM_abs.sum(axis=0)
    gt_P = CM_abs.sum(axis=1)
    true_P = np.diag(CM_abs)

    CM_iou = true_P / (pred_P + gt_P - true_P)
    miou = np.nanmean(CM_iou)

    if return_all:
        return CM_abs, CM_iou, miou
    else:
        return miou
