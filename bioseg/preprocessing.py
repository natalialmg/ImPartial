#Code adapted from csbdeep: http://csbdeep.bioimagecomputing.com/doc/_modules/csbdeep/data/generate.html#create_patches
#Code adapted from csbdeep: http://csbdeep.bioimagecomputing.com/doc/_modules/csbdeep/data/generate.html#create_patches

from tqdm import tqdm  # this tqdm seems to save temporarily in memory and then dispatch
import numpy as np
from csbdeep.data import norm_percentiles,no_background_patches,choice
# from BioSeg.preprocessing import *
from csbdeep.data import RawData
from skimage import io
from csbdeep.utils import normalize
from skimage.transform import rescale
from skimage.filters import rank

def generate_patches_syxc(raw_data, patch_size, n_patches_per_image,
                          patch_filter=no_background_patches(),
                          normalization = norm_percentiles(),
                          mask_filter_index = [0], fov_mask = None):

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
                #TODO: temporary patch, channel 0 is usually the denoised - this should be an option
                mask_filter = y[..., mask_filter_index]

                if len(mask_filter.shape)>2:
                    mask_filter = np.sum(mask_filter,axis=2)
                    # ndimage.

                    # selem_mean_filter = np.ones([patch_size[0], patch_size[1]])
                    # prob = rank.mean(mask_filter.astype('float'), selem=selem_mean_filter)
                    mask_filter[mask_filter>0] = 1
                # print('MASK FILTER :: ', np.sum(mask_filter),mask_filter_index, mask_filter.shape )

                # print(mask_filter[0:128, 0:128])
                patch_mask = patch_filter((mask_filter, x), (patch_size[0]*2-5,patch_size[1]*2-5))
                # print(patch_mask[0:128,0:128])
                # patch_mask if not provided (w x h); border slices, valid_inds (w and h)
            if fov_mask is not None:
                patch_mask = patch_mask*fov_mask

            border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[1].shape[0:2])])
            # print(patch_mask[border_slices])
            valid_inds = np.where(patch_mask[border_slices])
            n_valid = len(valid_inds[0])

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

    axes = 'SYXC'
    return X,Y,axes
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

