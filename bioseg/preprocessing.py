#Code adapted from csbdeep: http://csbdeep.bioimagecomputing.com/doc/_modules/csbdeep/data/generate.html#create_patches

from tqdm import tqdm  # this tqdm seems to save temporarily in memory and then dispatch
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.data import norm_percentiles,no_background_patches,choice
# from BioSeg.preprocessing import *
from csbdeep.data import RawData
from skimage import io
from csbdeep.utils import normalize
from skimage.transform import rescale

def generate_patches_syxc(raw_data, patch_size, n_patches_per_image,
                          patch_filter=no_background_patches(),
                          normalization = norm_percentiles(),
                          mask_filter_index = [0]):



    # TODO: Remake a normalization
    image_pairs, n_images = raw_data.generator(), raw_data.size
    n_patches = n_images * n_patches_per_image
    channel = 3

    i_valid = 0
    for i, (x, y, _axes, mask) in tqdm(enumerate(image_pairs), total=n_images):
        # print(i)
        if len(x.shape) < 3:
            x = x[:, :, np.newaxis]
        # print('shape :: ',y.shape)
        if len(y.shape) < 3:
            y = y[:, :, np.newaxis]


        if i == 0:
            X = np.empty((n_patches,) + tuple(patch_size) + tuple([x.shape[2]]), dtype=np.float32)
            Y = np.empty((n_patches,) + tuple(patch_size) + tuple([y.shape[2]]), dtype=np.float32)

        if len(y.shape) > 2:

            datas = (y, x)
            if patch_filter is None:
                patch_mask = np.ones(datas[0].shape[0:2], dtype=np.bool)
            else:
                #TODO: temporary parche, channel 0 is usually the denoised - this should be an option
                mask_filter = y[..., mask_filter_index]

                if len(mask_filter.shape)>2:
                    mask_filter = np.sum(mask_filter,axis=2)
                    mask_filter[mask_filter>0] = 1
                # print('MASK FILTER :: ', np.sum(mask_filter),mask_filter_index, mask_filter.shape )

                # print(mask_filter[0:128, 0:128])
                patch_mask = patch_filter((mask_filter, x), (patch_size[0]*2-5,patch_size[1]*2-5))
                # print(patch_mask[0:128,0:128])

                # patch_mask if not provided (w x h); border slices, valid_inds (w and h)
            border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[1].shape[0:2])])
            # print(patch_mask[border_slices])
            valid_inds = np.where(patch_mask[border_slices])
            n_valid = len(valid_inds[0])
            # print('nvalid :: ',n_valid)

            # print(border_slices)
            # print(n_valid)
            # print(datas[1].shape[0:2])
            # print()
            if n_valid > 0:
                # random choice of the valid indices
                sample_inds = choice(range(n_valid), n_patches_per_image, replace=(n_valid < n_patches_per_image))
                rand_inds = [v[sample_inds] + s.start for s, v in zip(border_slices, valid_inds)]

                ## Rand_inds has the indices start and end :)
                _Y, _X = [np.stack(
                    [data[tuple(slice(_r - (_p // 2), _r + _p - (_p // 2)) for _r, _p in zip(r, patch_size))] for r in
                     zip(*rand_inds)]) for data in datas]

                if normalization is not None:
                    _X, Y_aux = normalization(_X, np.expand_dims(_Y[..., 0], 3), x, np.expand_dims(y[..., 0], 2), None,
                                              channel)
                    _Y[..., 0] = Y_aux[..., 0]

                s = slice(i_valid * n_patches_per_image, (i_valid + 1) * n_patches_per_image)
                i_valid += 1
                X[s] = _X
                Y[s] = _Y

    if i_valid > 0:
        # print(i_valid,n_patches_per_image,i_valid*n_patches_per_image,X.shape)
        X = X[0:(i_valid + 1) * n_patches_per_image]
        Y = Y[0:(i_valid + 1) * n_patches_per_image]
    else:
        return None,None,None
            # ##plot##
            # plt.figure(figsize=(15, 10))
            # ix_plot = np.arange(n_patches_per_image)
            # np.random.shuffle(ix_plot)
            # print(ix_plot)
            #
            # for i in np.arange(5):
            #     plt.subplot(4, 5, i + 1)
            #     plt.imshow(X[ix_plot[i], :, :, 0])
            #
            # for i in np.arange(5):
            #     plt.subplot(4, 5, i + 6)
            #     plt.imshow(Y[ix_plot[i], :, :, 0])
            #
            # for i in np.arange(5):
            #     plt.subplot(4, 5, i + 11)
            #     plt.imshow(Y[ix_plot[i], :, :, 1])
            #
            # for i in np.arange(5):
            #     plt.subplot(4, 5, i + 16)
            #     plt.imshow(Y[ix_plot[i], :, :, 2])
            # plt.show()

    axes = 'SYXC'
    return X,Y,axes

def rescale_dataset(X, Y, rescale_value=[1.5], seed=1):
    X_res = np.array(X)
    Y_res = np.array(Y)
    if np.sum(np.abs(np.array(X.shape) - np.array(Y.shape))) > 0:
        print('Y and X must have same dimensions ')
        return

    ny = X.shape[1]
    nx = X.shape[2]

    np.random.seed(seed)
    if len(rescale_value) == 1:
        rescale_value = np.ones(X.shape[0]) * rescale_value
    else:
        rescale_value = np.random.uniform(size=X.shape[0]) * (rescale_value[1] - rescale_value[0]) + rescale_value[0]

    for i in np.arange(X_res.shape[0]):
        img_re = rescale(X[i, ...], rescale_value[i], anti_aliasing=False)
        img_re = img_re[int(img_re.shape[0] / 2) - int(ny / 2):int(img_re.shape[0] / 2) + int(ny / 2),
                 int(img_re.shape[1] / 2) - int(nx / 2):int(img_re.shape[1] / 2) + int(nx / 2)]
        X_res[i, ...] = img_re

        img_re = rescale(Y[i, ...], rescale_value[i], anti_aliasing=False)
        img_re = img_re[int(img_re.shape[0] / 2) - int(ny / 2):int(img_re.shape[0] / 2) + int(ny / 2),
                 int(img_re.shape[1] / 2) - int(nx / 2):int(img_re.shape[1] / 2) + int(nx / 2)]
        Y_res[i, ...] = img_re

    return X_res, Y_res

from scipy import ndimage
def compute_labels(prediction, threshold):
    pred_thresholded = prediction > threshold
    labels, _ = ndimage.label(pred_thresholded)
    return labels

def augment_data(X_train, Y_train):
    """
    Augments the data 8-fold by 90 degree rotations and flipping.

    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train_aug : array(float)
        Augmented array of training images.
    Y_train_aug : array(float)
        Augmented array of labelled training images.
    """
    X_ = X_train.copy()

    X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))

    Y_ = Y_train.copy()

    Y_train_aug = np.concatenate((Y_train, np.rot90(Y_, 1, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 2, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 3, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1)))

    # print('Raw image size after augmentation', X_train_aug.shape)
    # print('Mask size after augmentation', Y_train_aug.shape)

    return X_train_aug, Y_train_aug

def balance_augment_data(X, Y, augment=True, balance = True):
    ix_Y = np.sum(Y, axis=(1, 2, 3))
    X_train_wlabel = X[ix_Y > 0, ...]
    Y_train_wlabel = Y[ix_Y > 0, ...]
    X_train_nolabel = X[ix_Y == 0, ...]
    Y_train_nolabel = Y[ix_Y == 0, ...]

    if augment:
        X_aug, Y_aug = augment_data(X_train_wlabel, Y_train_wlabel)
        X_aug_nolabel, Y_aug_nolabel = augment_data(X_train_nolabel, Y_train_nolabel)
        del X_train_nolabel, Y_train_nolabel, X_train_wlabel, Y_train_wlabel
    else:
        X_aug, Y_aug = X_train_wlabel, Y_train_wlabel
        X_aug_nolabel, Y_aug_nolabel = X_train_nolabel, Y_train_nolabel

    if balance:
        ix_min = X_aug.shape[0]

        ix_shuffle = np.arange(X_aug.shape[0])
        np.random.shuffle(ix_shuffle)
        X_aug = X_aug[ix_shuffle[0:ix_min], ...]
        Y_aug = Y_aug[ix_shuffle[0:ix_min], ...]

        ix_shuffle = np.arange(X_aug_nolabel.shape[0])
        np.random.shuffle(ix_shuffle)
        X_aug_nolabel = X_aug_nolabel[ix_shuffle[0:ix_min], ...]
        Y_aug_nolabel = Y_aug_nolabel[ix_shuffle[0:ix_min], ...]

    # print('Changes made')
    X_aug = np.concatenate([X_aug, X_aug_nolabel], axis=0)
    Y_aug = np.concatenate([Y_aug, Y_aug_nolabel], axis=0)

    ix_shuffle = np.arange(X_aug.shape[0])
    np.random.shuffle(ix_shuffle)
    X_aug = X_aug[ix_shuffle, ...]
    Y_aug = Y_aug[ix_shuffle, ...]

    return X_aug, Y_aug

import re
def get_XY_patches_from_files_generic(pd_filter, tag_X_dir='input_dir',
                              tag_Y_dir='label_dir',
                              tag_X_file='input_file',
                              tag_Y_file='labels_file',
                              n_patches_per_image=50,
                              p_label=0.5,patch_size=(128, 128),
                              pmin = 1,pmax = 99.8, pnormalization = True):
    start = True
    for i in pd_filter.index.values:

        # print(i)
        pd_row = pd_filter.loc[i]

        # raw_image
        suffix = re.search('(?<=\.).*', pd_row[tag_X_file]).group()
        if suffix == 'npz':
            raw_image_in = np.load(pd_row[tag_X_dir] + pd_row[tag_X_file])
            raw_image_in = raw_image_in['image']
        else:
            raw_image_in = io.imread(pd_row[tag_X_dir] + pd_row[tag_X_file], as_gray=True)

        # raw_image_in = io.imread(pd_row[tag_X_dir] + pd_row[tag_X_file], as_gray=True)
        if pnormalization:
            raw_image_in = normalize(raw_image_in, pmin=pmin, pmax=pmax, clip=False)

        if len(raw_image_in.shape)>2:
            nchannels = raw_image_in.shape[-1]
        else:
            nchannels = 1

        # partial labels

        npz_file = np.load(pd_row[tag_Y_dir] + pd_row[tag_Y_file])

        # print(npz_file.files)
        for ch in np.arange(nchannels):
            if 'channel_'+str(ch) in npz_file.files:
                label_aux = npz_file['channel_'+str(ch)]
            else:
                label_aux = np.zeros([raw_image_in.shape[0],raw_image_in.shape[1],2])

            if ch == 0:
                label = np.array(label_aux)
            else:
                label = np.concatenate([label,label_aux],axis = -1)
            # print(label.shape)

        if 'instance' in npz_file.files:
            label_aux = npz_file['instance']
            label = np.concatenate([label, label_aux], axis=-1)

        raw_data = RawData.from_arrays(raw_image_in[np.newaxis,...], label[np.newaxis, ...])

        # print(pd_row[tag_X_dir] + pd_row[tag_X_file])
        # print(pd_row[tag_Y_dir] + pd_row[tag_Y_file])
        # print(np.sum(masks_gt))

        ## sample without filter for only labeled examples
        X_aux, Y_aux, axes = generate_patches_syxc(raw_data, patch_size,
                                                   n_patches_per_image,
                                                   normalization=None, patch_filter=None)

        n_patches_with_labels = np.sum(np.sum(Y_aux,axis = (1,2,3))>0)

        n_patches_with_labels_add = (n_patches_per_image*p_label - n_patches_with_labels)/(1-p_label)
        n_patches_with_labels_add = int(n_patches_with_labels_add)

        # print(n_patches_with_labels)
        # print(X_aux.shape,Y_aux.shape)
        # print(n_patches_with_labels_add)

        if n_patches_with_labels_add > 0:
            X_labeled_aux, Y_labeled_aux, axes = generate_patches_syxc(raw_data, patch_size,
                                                                       n_patches_with_labels_add,
                                                                       normalization=None,
                                                                       mask_filter_index=np.arange(label.shape[-1]))

            # print(np.sum(np.sum(masks_gt[:, :, np.arange(masks_gt.shape[-1])],axis = -1)))
            # print(X_labeled_aux is None)
            X_aux = np.concatenate([X_aux, X_labeled_aux], axis=0)
            Y_aux = np.concatenate([Y_aux, Y_labeled_aux], axis=0)

        # print(np.arange(masks_gt.shape[-1]))
        # print(X_aux.shape, Y_aux.shape)
        if X_aux is not None:
            if start:
                X = X_aux
                Y = Y_aux
                # ix_list = [i for j in np.arange(X.shape[0])]
                start = False
            else:
                X = np.concatenate([X, X_aux], axis=0)
                Y = np.concatenate([Y, Y_aux], axis=0)
                # ix_list.extend([i for j in np.arange(X.shape[0])])
                #break

        # print(X_aux.shape[0],
        #       np.sum(np.sum(Y_aux,axis = (1,2,3))>0),
        #       np.sum(np.sum(Y_aux,axis = (1,2,3))>0)/X_aux.shape[0])

    return X, Y
