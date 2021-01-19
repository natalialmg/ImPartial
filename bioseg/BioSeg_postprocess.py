import numpy as np
import pandas as pd
from csbdeep.utils import normalize
import skimage.measure as measure
from sklearn.metrics import roc_auc_score
import skimage.morphology as morphology
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.metrics import f1_score,accuracy_score,jaccard_score

def get_Yest_list(Y_est, tag='instance_0', ix_fg_list=[0],
                  pnorm=True, tipo=0, pmin=0.01, pmax=99.9):
    Y_est_list = []

    if tipo == 0:

        for i in np.arange(len(Y_est)):

            start = True
            for ix_fg in ix_fg_list:
                if start:
                    Y_est_i = Y_est[i][tag][:, :, :, ix_fg]
                else:
                    Y_est_i += Y_est[i][tag][:, :, :, ix_fg]
                start = False

            if (Y_est[i][tag].shape[0] == 1):
                Y_est_i = Y_est[i][tag][0, ...]

            if pnorm:
                Y_est_i = normalize(Y_est_i, pmin=pmin, pmax=pmax, clip=True)
            Y_est_list.append(Y_est_i)
    else:
        Y_est_aux = Y_est[tag]
        for i in range(Y_est_aux.shape[0]):
            start = True
            for ix_fg in ix_fg_list:
                if start:
                    Y_est_i = Y_est_aux[i, :, :, ix_fg]
                else:
                    Y_est_i += Y_est_aux[i, :, :, ix_fg]
                start = False

            if pnorm:
                Y_est_i = normalize(Y_est_i, pmin=pmin, pmax=pmax, clip=True)
            Y_est_list.append(Y_est_i)

    return Y_est_list

def single_channel_evaluation(pd_data, model_evaluation, tag='channel_0',
                              nbatch=4,group='test',ix_fg_list=[0], ix_label = None):
    ## Get list of data ##
    X_list = None
    Y_gt_list = []
    prefix_list = []

    for i in range(len(pd_data)):
        group_i = pd_data['group'][i]
        if group_i == group:

            npz_read = np.load(pd_data['input_dir'][i] + pd_data['input_file'][i])
            image = npz_read['image']
            if ix_label is None:
                label = npz_read['label']
            else:
                label = npz_read['label'][...,ix_label]
            prefix_list.append([pd_data['prefix'][i]])

            if X_list is None:
                X_list = np.array(image)[np.newaxis, ...]
            else:
                X_list = np.concatenate([X_list, image[np.newaxis, ...]])
            Y_gt_list.append(label)

    ### Estimation ###
    ix = 0
    while ix < X_list.shape[0]:
        nmax = np.minimum(ix + nbatch, X_list.shape[0])
        # print(X_list.shape)
        print('Evaluation : ', group, '; from sample ' ,ix, ' to ',nmax)
        if (len(X_list.shape) <= 3):
            Y_est_aux = model_evaluation(X_list[ix:nmax, ...][...,np.newaxis])
        else:
            Y_est_aux = model_evaluation(X_list[ix:nmax, ...])

        if ix == 0:
            Y_est = Y_est_aux.copy()
        else:
            for key in Y_est.keys():
                Y_est[key] = np.concatenate([Y_est[key], Y_est_aux[key]], axis=0)
        ix += nbatch

    Y_est_train_list = get_Yest_list(Y_est, tag=tag, tipo=1, ix_fg_list=ix_fg_list)

    return Y_est_train_list, Y_gt_list, X_list,prefix_list


# Dice Score & IoU for binary segmentation
class Evaluator(object):
    def __init__(self):
        self.Dice = 0
        self.IoU = 0
        self.AIJ = 0
        self.num_batch = 0
        self.eps = 1e-4

    def dice_fn(self, gt_image, pre_image):
        eps = 1e-4
        batch_size = pre_image.shape[0]

        pre_image = pre_image.reshape(batch_size, -1).astype(np.bool)
        gt_image = gt_image.reshape(batch_size, -1).astype(np.bool)

        intersection = np.logical_and(pre_image, gt_image).sum(axis=1)
        union = pre_image.sum(axis=1) + gt_image.sum(axis=1) + eps
        Dice = ((2. * intersection + eps) / union).mean()
        IoU = Dice / (2. - Dice)
        return Dice, IoU


    def mdice_fn(self, target, pred):
        # This function is from :: https://github.com/naivete5656/WSISPDR/blob/master/utils/for_review.py
        '''
        :param target: hxw label
        :param pred: hxw label
        :return: mIoU, mDice
        '''
        iou_mean = 0.
        dice_mean = 0.
        labels_matched = []
        inter_all = 0.
        union_all = 0.
        unmatch_all = 0.

        for idx, target_label in enumerate(range(1, target.max() + 1)):
            if np.sum(target == target_label) < 20:
                target[target == target_label] = 0
                # seek pred label correspond to the label of target
            correspond_labels = pred[target == target_label]
            correspond_labels = correspond_labels[correspond_labels != 0]
            unique, counts = np.unique(correspond_labels, return_counts=True)
            try:
                max_label = unique[counts.argmax()]
                pred_mask = np.zeros(pred.shape)
                pred_mask[pred == max_label] = 1
                labels_matched.append(max_label)
            except ValueError:
                bou_list = []
                max_bou = target.shape[0]
                max_bou_h = target.shape[1]
                bou_list.extend(target[0, :])
                bou_list.extend(target[max_bou - 1, :])
                bou_list.extend(target[:, max_bou_h - 1])
                bou_list.extend(target[:, 0])
                bou_list = np.unique(bou_list)
                for x in bou_list:
                    target[target == x] = 0
                pred_mask = np.zeros(pred.shape)

            # create mask
            target_mask = np.zeros(target.shape)
            target_mask[target == target_label] = 1
            pred_mask = pred_mask.flatten()
            target_mask = target_mask.flatten()

            tp = pred_mask.dot(target_mask)
            fn = pred_mask.sum() - tp
            fp = target_mask.sum() - tp
            inter_all += tp
            union_all += tp + fp + fn

            iou = ((tp + self.eps) / (tp + fp + fn + self.eps))
            dice = (2 * tp + self.eps) / (2 * tp + fn + fp + self.eps)
            iou_mean = (iou_mean * idx + iou) / (idx + 1)
            dice_mean = (dice_mean * idx + dice) / (idx + 1)

        for i in np.unique(pred):
            if (i != 0) & (i not in labels_matched):
                unmatch_all += np.sum(pred==i)
        aij = inter_all/(union_all+unmatch_all)

        return dice_mean, iou_mean, aij

    def add_pred(self, gt_image, pre_image):
        pre_image = measure.label(pre_image)
        pre_image = morphology.remove_small_objects(pre_image, min_size=20)
        pre_image = binary_fill_holes(pre_image > 0)
        pre_image = measure.label(pre_image)

        batch_mdice, batch_miou, batch_aij = self.mdice_fn(gt_image, pre_image)
        self.Dice = (self.Dice * self.num_batch + batch_mdice) / (self.num_batch + 1)
        self.IoU = (self.IoU * self.num_batch + batch_miou) / (self.num_batch + 1)
        self.AIJ = (self.AIJ * self.num_batch + batch_aij) / (self.num_batch + 1)
        self.num_batch += 1

    def reset(self):
        self.Dice = 0
        self.IoU = 0
        self.AIJ = 0
        self.num_batch = 0


def single_eval(label_gt,mask_th):
    performance = Evaluator()
    performance.add_pred(label_gt,mask_th)
    return [performance.IoU,performance.Dice,performance.AIJ]

def evaluate_thresholds(label_gt, prob_est, th_list=None):
    if th_list is None:
        th_list = np.linspace(0, 1, 21)[1:-1]

    values = []
    for th in th_list:
        mask_th = np.zeros_like(prob_est)
        mask_th[prob_est >= th] = 1
        IoU, Dice, AIJ = single_eval(label_gt, mask_th)

        mask_gt = np.zeros_like(label_gt)
        mask_gt[label_gt>0] = 1
        F1 = f1_score(mask_gt.flatten(),mask_th.flatten())
        Jacc = jaccard_score(mask_gt.flatten(),mask_th.flatten())
        Acc = accuracy_score(mask_gt.flatten(), mask_th.flatten())


        values.append([th, IoU, Dice, AIJ, F1, Jacc, Acc])
    return values

def get_performance(Y_est_plot, Y_gt_plot, th_list=None):
    stats = None
    print('Evaluation i/nsamples')
    for i in range(len(Y_est_plot)):
        print(i,len(Y_est_plot))
        mask_gt = np.zeros_like(Y_gt_plot[i])
        mask_gt[Y_gt_plot[i] > 0] = 1.0
        label_gt = measure.label(mask_gt)

        auc = roc_auc_score(y_true=1 - mask_gt.flatten(), y_score=1 - Y_est_plot[i].flatten())

        values = evaluate_thresholds(label_gt, Y_est_plot[i], th_list=th_list)
        values = np.array(values)
        values = np.concatenate([values, (np.ones([values.shape[0]]) * auc)[:, np.newaxis]], axis=1)

        stats_i = np.concatenate([values, (np.ones([values.shape[0]]) * i)[:, np.newaxis]], axis=1)
        if stats is None:
            stats = np.array(stats_i)
        else:
            stats = np.concatenate([stats, stats_i], axis=0)

    pd_stats = pd.DataFrame(stats, columns=['th', 'IoU', 'Dice', 'AIJ', 'F1','Jacc','Acc','auc', 'index'])
    return pd_stats


def get_best(pd_stats,metric_list = ['IoU', 'Dice', 'AIJ','F1',	'Jacc',	'Acc','auc']):
    best_values = []
    for index in pd_stats['index'].unique():

        best_aux = []
        pd_loc = pd_stats.loc[pd_stats['index'] == index]
        #     patch = pd_loc['patch'].unique()[0]
        #     prefix = pd_loc['prefix'].unique()[0]
        #     pd_loc = pd_stats_train.loc[(pd_stats_train.patch == patch) & (pd_stats_train.prefix == prefix)]

        #         print(index,patch,prefix)
        for metric in metric_list:
            best_aux.append(pd_loc[metric].max())
        best_aux.append(pd_loc['group'].unique()[0])
        best_aux.append(pd_loc['prefix'].unique()[0])
        best_values.append(best_aux)

    metric_list.extend(['group', 'prefix'])
    pd_best = pd.DataFrame(data=best_values, columns=metric_list)
    return pd_best


def get_best_th(pd_stats,metric_list = ['IoU', 'Dice', 'AIJ','F1',	'Jacc',	'Acc']):
    best_value = {}
    best_th = {}

    for th in pd_stats['th'].unique():
        pd_loc = pd_stats.loc[pd_stats['th'] == th]
        for metric in metric_list:
            mvalue = pd_loc[metric].mean()
            if metric in best_value.keys():
                if best_value[metric] < mvalue:
                    best_value[metric] = mvalue
                    best_th[metric] = th
            else:
                best_value[metric] = mvalue
                best_th[metric] = th
    return best_value, best_th


def get_stats_th(pd_stats, best_th,metric_list = ['IoU', 'Dice', 'AIJ','F1','Jacc',	'Acc']):
    best_values = []
    for index in pd_stats['index'].unique():

        best_aux = []
        pd_loc = pd_stats.loc[pd_stats['index'] == index]
        #     patch = pd_loc['patch'].unique()[0]
        #     prefix = pd_loc['prefix'].unique()[0]
        #     pd_loc = pd_stats_train.loc[(pd_stats_train.patch == patch) & (pd_stats_train.prefix == prefix)]

        #         print(index,patch,prefix)
        for metric in metric_list:
            th_metric = best_th[metric]
            pd_filter = pd_loc.loc[pd_loc['th'] == th_metric]
            best_aux.append(pd_filter[metric].mean())
        best_aux.append(pd_loc['auc'].mean())
        best_aux.append(pd_loc['group'].unique()[0])
        best_aux.append(pd_loc['prefix'].unique()[0])

        best_values.append(best_aux)
    metric_list.extend(['auc', 'group', 'prefix'])
    pd_best = pd.DataFrame(data=best_values, columns=metric_list)
    return pd_best


def get_seg_image(X,Y,th=0,min_size = 20):
    Y_plot = np.zeros_like(Y)
    Y_plot[Y>th]=1
    Y_plot = morphology.remove_small_objects(Y_plot.astype('bool'), min_size=min_size)
    Y_plot = binary_fill_holes(Y_plot > 0)
    Y_plot = Y_plot.astype('int')
    Y_plot = morphology.dilation(Y_plot, selem=morphology.disk(1)) - Y_plot

    implot = np.zeros([X.shape[0], X.shape[1], 3])
    # print(X.shape, Y_plot.shape)
    implot[..., :] = (normalize(X, pmin=1, pmax=99, clip=True) * 0.9 *(1-Y_plot))[...,np.newaxis]
    implot[..., 0] += Y_plot
    # implot[..., 1] = (1 - Y_plot) * implot[..., 1]

    return implot



