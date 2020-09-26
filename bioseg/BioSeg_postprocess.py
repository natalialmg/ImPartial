import numpy as np
from BioSeg_models import set_bioseg_model
import tensorflow as tf
from csbdeep.utils import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def backfore_performance(z_gt,z_est,mask = None):
    #z_gt and z_est are binary matrices, same shape

    if mask is not None:
        z_est = z_est[mask>0]
        z_gt = z_gt[mask>0]

    jacc_fore = np.sum(z_gt*z_est>0)/np.sum(z_est+z_gt>0)
    jacc_back = np.sum((1-z_gt) * (1-z_est) > 0) / np.sum((1-z_est) + (1-z_gt) > 0)
    TPR = np.sum(z_gt*z_est)/np.sum(z_gt)
    TNR = np.sum((1-z_gt)*(1-z_est))/np.sum(1-z_gt)
    FPR = np.sum((1-z_gt)*(z_est))/np.sum(1-z_gt)
    FNR = np.sum((z_gt)*(1-z_est)) / np.sum(z_gt)

    return [jacc_fore,jacc_back,TPR,TNR,FPR,FNR]

def get_mask_best_th(y_pred, y_gt, stat='accuracy', th_list=np.linspace(0.01, 0.99, 20)):

    best_value = 0
    best_th = 0

    if stat is 'accuracy':
        stat_func = accuracy_score

    if stat is 'precision':
        stat_func = precision_score

    if stat is 'recall':
        stat_func = recall_score

    if stat is 'f1':
        stat_func = f1_score

    patience = 0
    for th in th_list:
        z_mask = np.array(y_pred)
        z_mask = z_mask > th
        #     jacc_fore,jacc_back,TPR,TNR,FPR,FNR = backfore_performance(filter_image,z_mask,mask = None)
        #     print(jacc_fore,jacc_back,TPR,TNR,FPR,FNR)

        score = stat_func(y_gt.flatten(), z_mask.flatten())
        if score >= best_value:
            best_value = score
            best_th = th
            patience = 0

        else:
            patience += 1
            if (patience == 2):
                break
    return best_th
    #     acc_score = accuracy_score
    #     prec_score = precision_score(y_gt.flatten(), z_mask.flatten())
    #     rec_score = recall_score(y_gt.flatten(), z_mask.flatten())
    #     f1 = f1_score(y_gt.flatten(), z_mask.flatten())
    #     rows.append([th, acc_score, prec_score, rec_score, f1])
    # # print(acc_score,prec_score,rec_score,f1)
    #
    # pd_stats = pd.DataFrame(data=rows, columns=['th', 'accuracy', 'precision', 'recall', 'f1'])
    # stat_val = pd_stats[stat].values
    # ix_val = np.argmax(stat_val)
    # th_final = pd_stats['th'].values[ix_val]
    # #     print(th_final)
    # print(pd_stats.iloc[ix_val])


def get_FN(estimated_mask, labels_image, threshold=0.5):
    ## Get FN labels considering a threshold on the inclusion on the obtained mask

    FN = np.zeros_like(labels_image)
    labels_unique = np.unique(labels_image)
    for label in labels_unique:
        if label != 0:  # 0 is assumed to be background
            aux = np.zeros_like(labels_image)
            aux[labels_image == label] = 1

            intersection = aux * estimated_mask
            area_ratio = np.sum(intersection) / np.sum(aux)
            # print(area_ratio)
            if area_ratio < threshold:
                FN += aux-intersection.astype('int')
    return FN

def get_model_dic(save_data_full,key_names):
    ix = 1
    model_dic = {}
    for key in key_names:
        save_data = save_data_full[key]
        input_dic = save_data['input_dic']
        input_dic['load_tf_file'] = save_data['model_file']
        print(key)
        X = np.random.random([26,128,128,1])
        model_dic[key],_ = set_bioseg_model([X], input_dic, train = False)
        # print()
        # print(model_dic)
        model_dic[key].config.means=save_data['more_specs']['in_mean_std'][0]
        model_dic[key].config.stds=save_data['more_specs']['in_mean_std'][1]
        ix += 1
    return model_dic

def clip(array,vmin,vmax):
    return np.minimum(np.maximum(array,vmin),vmax)

def get_demix_prediction(config, y_pred):
    nfore = config.n_fore_modes
    nback = config.n_back_modes
    nchannels = config.n_channel_in
    ncomponents = config.n_components
    ninstance = config.n_instance_seg
    patch_size = (y_pred.shape[1],y_pred.shape[2])

    ix = 0
    foreback_pred = y_pred[..., ix:ix + (nfore + nback) * nchannels * ncomponents]
    # print(foreback_pred.shape)

    ix += (nfore + nback) * nchannels * ncomponents
    instance_pred = y_pred[..., ix:ix + ninstance * 2]

    ## Foreback mask !!
    foreback_pred = tf.reshape(foreback_pred, [-1, patch_size[0], patch_size[1],
                                               nchannels, nfore + nback, ncomponents])
    foreback_mask = tf.nn.softmax(foreback_pred[..., 0], axis=-1)
    # foreback_mask = tf.stack([(tf.reduce_sum(foreback_mask[..., 0:nfore], -1)),
    #                           (tf.reduce_sum(foreback_mask[..., nfore:], -1))],
    #                          axis=-1)
    foreback_mask = foreback_mask.eval()
    foreback_mask = np.reshape(foreback_mask, (foreback_mask.shape[0],
                                               foreback_mask.shape[1],
                                               foreback_mask.shape[2], -1))

    ## Instance mask !!
    instance_pred = tf.reshape(instance_pred, [-1, patch_size[0], patch_size[1],
                                               ninstance, 2])
    instance_mask = tf.nn.softmax(instance_pred, axis=-1)
    instance_mask = instance_mask.eval()
    instance_mask = np.reshape(instance_mask, (instance_mask.shape[0],
                                               instance_mask.shape[1],
                                               instance_mask.shape[2], -1))

    print('foreback/Instance mask shape : ',foreback_mask.shape,instance_mask.shape)

    return np.concatenate([foreback_mask, instance_mask], axis=-1)

def get_out_model(model, raw_image_in):

    # std
    if model.config.normalizer == 'std':
        mean_o = float(model.config.means[0])
        std_o = float(model.config.stds[0])
    else:
        mean_o = 0.0
        std_o = 1.0
        # print('No STD')

    # output_image
    # print(mean_o,std_o)

    if len(raw_image_in.shape)>2:
        y_pred = model.predict(raw_image_in, 'YXC')
    else:
        y_pred = model.predict((raw_image_in), 'YX')

    if len(y_pred.shape) <= 3:
        y_pred = y_pred[np.newaxis,...]
    # print('y_pred :: ' ,y_pred.shape)
    out_mask = get_demix_prediction(model.config, y_pred)

    means_out = {}
    stds_out = {}
    # for i in range(out_model.shape[-2]):
    #     means_out[i] = out_model[..., i,1].eval() * std_o + mean_o
    #     # print(out_model[..., i,1])
    #     stds_out[i] = np.exp(clip(out_model[..., i,2].eval(), -5, 5)) * std_o + mean_o
    #
    # # if means == 2:
    # #     means_out[0] = out_model[..., 1] * std_o + mean_o
    # #     means_out[1] = (np.maximum(out_model[..., 0], 0) + out_model[..., 1]) * std_o + mean_o
    # #     stds_out[0] = np.exp(clip(out_model[..., 4], -5, 5)) * std_o + mean_o
    # #     stds_out[1] = np.exp(clip(out_model[..., 3], -5, 5)) * std_o + mean_o
    # # else:
    # #     means_out[0] = out_model[..., 0] * std_o + mean_o
    # #     means_out[1] = out_model[..., 1] * std_o + mean_o
    # #     stds_out[0] = np.exp(clip(out_model[..., 4], -5, 5)) * std_o + mean_o
    #     stds_out[1] = np.exp(
    #         clip(np.maximum(out_model[..., 3], 0) + clip(out_model[..., 4], -5, 5), -5, 5)) * std_o + mean_o

    return out_mask, means_out, stds_out


from scipy import ndimage
def compute_labels(prediction, threshold):
    pred_thresholded = prediction > threshold
    labels, _ = ndimage.label(pred_thresholded)
    return labels


def predict_label_masks(Y_est, Y, threshold, measure):
    predicted_images = []
    precision_result = []
    for i in range(len(Y)):
        if (np.max(Y[i]) == 0 and np.min(Y[i]) == 0):
            continue
        else:
            labels = compute_labels(Y_est[i], threshold)

            #             plt.imshow(labels)
            #             plt.show()

            #             plt.imshow(Y[i])
            #             plt.show()
            tmp_score = measure(Y[i], labels)
            predicted_images.append(labels)
            precision_result.append(tmp_score)
    return predicted_images, np.mean(precision_result)


def optimize_thresholds(Y_est, Y_val, measure,verbose = True):
    """
     Computes average precision (AP) at different probability thresholds on validation data and returns the best-performing threshold.

     Parameters
     ----------
     Y_est : array(float)
         Array of estimated validation images.
     Y_val : array(float)
         Array of validation labels
     model: keras model

     mode: 'none', 'StarDist'
         If `none`, consider a U-net type model, else, considers a `StarDist` type model
     Returns
     -------
     computed_threshold: float
         Best-performing threshold that gives the highest AP.
     """
    if verbose:
        print('Computing best threshold: ')
    precision_scores = []

    for ts in (np.linspace(0.1, 1, 19)):
        _, score = predict_label_masks(Y_est, Y_val, ts, measure)
        precision_scores.append((ts, score))
        if verbose:
            print('Score for threshold =', "{:.2f}".format(ts), 'is', "{:.4f}".format(score))

    sorted_score = sorted(precision_scores, key=lambda tup: tup[1])[-1]
    computed_threshold = sorted_score[0]
    best_score = sorted_score[1]
    return computed_threshold, best_score


import os

def model_evaluation_fc(input_dic, perc_normalization=True, pmin=1, pmax=99.8):
    X = np.random.random([5, 128, 128, 1])

    # basedir = input_dic['basedir']
    # model_name = input_dic['model_name']
    # if os.path.exists(basedir+'/'+model_name+'/config.json'):
    #     print('Normalization : ', model.config.normalizer)
    #     config_dict = load_json(basedir + '/' + model_name + '/config.json')
    #     model.config.means = config_dict['means']
    #     model.config.stds = config_dict['stds']

    model, _ = set_bioseg_model([X], input_dic, train=False)

    # print(' CONFIG ::: ')
    # print(model.config)
    model_dic = {}
    model_dic[0] = model

    perc_normalization = perc_normalization
    pmin = pmin
    pmax = pmax

    def model_evaluation(raw_image):
        if perc_normalization:
            raw_image_in = normalize(raw_image, pmin=pmin, pmax=pmax, clip=False)
        else:
            raw_image_in = raw_image+0

        ### Evaluate Model ###
        cont = 0
        for key in model_dic.keys():
            if cont == 0:
                out_mask, means_out, stds_out = get_out_model(model_dic[key], raw_image_in)
                #             out_mask = (out_mask - np.min(out_mask))/(np.max(out_mask)-np.min(out_mask))
            else:
                out_mask_aux, means_out, stds_out = get_out_model(model_dic[key], raw_image_in)
                #             out_mask_aux = (out_mask_aux - np.min(out_mask_aux))/(np.max(out_mask_aux)-np.min(out_mask_aux))
                out_mask += out_mask_aux
            cont += 1
        out_mask /= cont + 1

        return out_mask

    return model_evaluation

def get_best_threshold(Y_est_train, Y_fg_label_train, Y_bg_label_train, th_bins=21):
    iou_means = []
    th_list = np.linspace(0, 1, th_bins)
    for th in th_list:

        iou_list = []
        for i in range(len(Y_est_train)):
            Y_est = Y_est_train[i]
            Y_fg = Y_fg_label_train[i]
            Y_bg = Y_bg_label_train[i]

            Y_est_th = np.zeros_like(Y_est)
            Y_est_th[Y_est >= th] = 1

            iou = np.sum(Y_est_th * Y_fg) + np.sum((1 - Y_est_th) * Y_bg)
            iou = iou / np.sum(Y_fg + Y_bg)
            iou_list.append(iou)
        iou_list = np.array(iou_list)
        # print(th, iou_list.mean())
        iou_means.append(iou_list.mean())

    iou_means = np.array(iou_means)
    ix = np.argmax(iou_means)
    threshold = th_list[ix]
    iou_best = iou_means[ix]

    print('Best : ', threshold, iou_best)

    return threshold, iou_best