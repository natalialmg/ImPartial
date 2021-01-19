import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def loss_demix(relative_weights = None, type_loss = 'full',
               type_rec = 'gauss', patch_size = (128,128),
               nback = 1, nfore = 1,
               nbacki = 1, nforei = 1, nchannels = 1, ninstance = 1,
               std = False, mean = True):

    C_laplace = np.log((2.0))
    C_gauss = np.log(np.sqrt(2.0*np.pi))

    patch_size = eval(patch_size)
    ninstance = int(ninstance)
    nback = int(nback) #back components
    nfore = int(nfore)  # fore components
    nbacki = int(nbacki)  # back components
    nforei = int(nforei)  # fore components
    nchannels = int(nchannels)
    std = (std == 'True')
    mean = (mean == 'True')

    ## Relative weights on objectives
    if relative_weights is None:
        relative_weights = [1.0, 0.0, 0.0, 0.0]
    class_weights = tf.constant([relative_weights])

    def gaussian_nll(target,mean,std_log):
        return K.square((target - mean) / tf.exp(std_log))/2 + std_log + C_gauss
    def laplace_nll(target,mean,std_log):
        return K.abs(target - mean) / tf.exp(std_log) + std_log + C_laplace

    ## Reconstruction losses
    if type_rec is 'gauss':
        rec_loss = gaussian_nll
    if type_rec is 'laplace':
        rec_loss = laplace_nll


    mask_fore = np.zeros([nback+nfore])
    mask_fore[0:nfore]=1
    mask_fore = tf.convert_to_tensor(mask_fore[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:], dtype=tf.float32)
    # print(mask_fore.shape)
    ksize = 3
    kernel_in = np.zeros([ksize, ksize, (2) * nchannels, (2) * nchannels])
    for i in np.arange((2) * nchannels):
        # kernel_in[:,:,i,i] = 1/(ksize*ksize)
        kernel_in[:, :, i, i] = -1
        kernel_in[int(np.floor(ksize/2)), int(np.floor(ksize/2)), i, i] = ksize*ksize-1
    kernel_in = tf.constant(kernel_in, dtype=tf.float32)
    # kernel_in = tf.constant(kernel_in/(ksize*ksize), dtype=tf.float32)

    # ksize = 5
    # kernel_in_l = np.zeros([ksize, ksize, (nfore + nback) * nchannels, (nfore + nback) * nchannels])
    # for i in np.arange((nfore + nback) * nchannels):
    #     # kernel_in[:,:,i,i] = 1/(ksize*ksize)
    #     kernel_in_l[:, :, i, i] = -1
    #     kernel_in_l[int(np.floor(ksize/2)), int(np.floor(ksize/2)), i, i] = ksize*ksize-1
    # kernel_in_l = tf.constant(kernel_in_l, dtype=tf.float32)

    if std & mean:
        ncomponents = 3 #mean std and mask
    else:
        ncomponents = 2 #only (mean or std) and mask

    def get_gt(y_true):
        #ytrue : #batchxpatch_size[0] x patch_size[1] x (nchannels + 1 + nchannelsx2 + ninstancesx2)

        ix = 0
        rec_gt = y_true[..., ix:ix+nchannels]
        ix += nchannels
        n2v_mask = y_true[..., ix:ix+nchannels]
        ix += nchannels
        foreback_gt = y_true[..., ix:ix+(nchannels * 2)]
        ix += (nchannels * 2)
        if ninstance > 0:
            instance_gt = y_true[..., ix: ix+(ninstance * 2):]
        else:
            instance_gt = 0.0



        return rec_gt,n2v_mask,foreback_gt,instance_gt

    def couple_foreback(y):
        #only nonegative values for foreground, & foreground components bigger than average foreground
        y = tf.nn.relu(y)*mask_fore + y*(1-mask_fore) +\
            tf.expand_dims(tf.math.reduce_sum(y*(1-mask_fore)/nback,axis=-1),axis = -1)*mask_fore
        return y

    def get_std(y):
        y = tf.clip_by_value(y, -5, 5)
        y_log_std = couple_foreback(y[...,0])
        y_log_std = tf.math.reduce_mean(y_log_std , axis = [1,2])  #nbatch x nchannel x (nfore + nback)
        y_log_std = tf.expand_dims(tf.expand_dims(y_log_std, axis=1), axis=1)  # nbatch x 1 x 1 x nchannel x (nfore + nback)
        return 0.0,y_log_std

    def get_mean(y):
        y_mean = couple_foreback(y[...,0])
        y_mean = tf.math.reduce_mean(y_mean, axis=[1, 2])  #nbatch x nchannel x (nfore + nback)
        y_mean = tf.expand_dims(tf.expand_dims(y_mean, axis=1), axis=1) #nbatch x 1 x 1 x nchannel x (nfore + nback)

        return y_mean,0.0

    def get_mean_std(y):
        y_mean = couple_foreback(y[...,0]) # nbatch x patch x patch x nchannels x (nfore + nback)
        y_mean = tf.math.reduce_mean(y_mean, axis=[1, 2])  #nbatch x nchannel x (nfore + nback)
        y_mean = tf.expand_dims(tf.expand_dims(y_mean, axis=1), axis=1)  # nbatch x 1 x 1 x nchannel x (nfore + nback)

        ## single std..
        y_log_std = tf.clip_by_value(y[...,1], -5, 5)
        # y_log_std = couple_foreback(y_log_std)
        y_log_std = tf.math.reduce_mean(y_log_std, axis=[1, 2])  #nbatch x nchannel x (nfore + nback)
        y_log_std = tf.expand_dims(tf.expand_dims(y_log_std, axis=1),axis=1)  # nbatch x 1 x 1 x nchannel x (nfore + nback)

        return y_mean,y_log_std

    if mean & std :
        mean_std = get_mean_std
    if mean & (not std):
        mean_std = get_mean
    if (not mean) & std:
        mean_std = get_std

    # gamma = 0.4
    def bs_loss(onehot_gt,onehot_pred):
        loss = tf.reduce_sum(onehot_gt*(onehot_gt-onehot_pred)**2,axis = 0)/tf.math.maximum(tf.reduce_sum(onehot_gt,axis=0),1) # size = 2
        return loss

#### with instance ####


    def get_estimation(y_pred):
        ix = 0
        foreback_pred = y_pred[...,ix:ix+(nfore+nback)*nchannels*ncomponents]
        if ninstance == 0:
            return foreback_pred
        else:
            ix += (nfore+nback)*nchannels*ncomponents
            instance_pred = y_pred[..., ix:ix + ninstance*(nforei+nbacki)]
            return foreback_pred,instance_pred

    def demix(y_true, y_pred): ## Laplace mean = 0

        ## Get gt inputs
        rec_gt, n2v_mask, foreback_gt, instance_gt = get_gt(y_true)

        ## Get predictions
        if ninstance > 0:
            foreback_pred, instance_pred = get_estimation(y_pred)
        else:
            foreback_pred = get_estimation(y_pred)

        ## Foreground- background reconstruction loss
        foreback_pred = tf.reshape(foreback_pred, [-1, patch_size[0], patch_size[1], nchannels, nfore + nback, ncomponents])
        foreback_mask = tf.nn.softmax(foreback_pred[..., 0], axis=-1)  # nbatch x patch x patch x nchannels x (nfore + nback)
        y_mean, y_log_std = mean_std(foreback_pred[..., 1:]) # nbatch x 1 x 1 x nchannels x (nfore + nback)  [both same size or 0.0]

        loss_denoise = rec_loss(tf.expand_dims(rec_gt, axis=-1), y_mean, y_log_std) * foreback_mask  # nbatch x w x h x nchannels x (nfore + nback)
        loss_denoise = tf.reduce_sum(loss_denoise, -1) # nbatch x w x h x nchannels
        loss_denoise = tf.reduce_sum(loss_denoise * n2v_mask) / tf.math.maximum(tf.reduce_sum(n2v_mask),1)# + \
                       #0.1 * tf.reduce_mean(tf.reduce_sum(foreback_mask_aux ** 2, -1)) #+ \
                       # 0.5 * tf.reduce_mean(tf.reduce_sum(y_mean_aux ** 2, -1))#+ \
                       # 0.5 * tf.reduce_mean(tf.reduce_sum(y_log_aux ** 2, -1))

        # annotations foreback loss
        onehot_gt = tf.reshape(foreback_gt, [-1, 2])  # (nbatch x w x h x nchannels) X 2
        onehot_pred = tf.reshape(tf.stack([(tf.reduce_sum(foreback_mask[..., 0:nfore], -1)),
                          (tf.reduce_sum(foreback_mask[..., nfore:], -1))],
                                          axis=-1), [-1, 2]) # (nbatch x w x h x nchannels) X 2
        loss_foreback = bs_loss(onehot_gt, onehot_pred) # size 2

        # instance loss
        if ninstance > 0:
            instance_pred = tf.reshape(instance_pred, [-1, patch_size[0], patch_size[1], ninstance, nforei+nbacki])
            instance_pred = tf.nn.softmax(instance_pred,axis=-1)  # nbatch x patch x patch x ninstance X 2

            onehot_instance_gt = tf.reshape(instance_gt, [-1, 2]) # (nbatch x patch x patch x ninstance) X 2
            onehot_instance_pred = tf.reshape(tf.stack([(tf.reduce_sum(instance_pred[..., 0:nforei], -1)),
                          (tf.reduce_sum(instance_pred[..., nforei:], -1))],
                                          axis=-1), [-1, 2])  # (nbatch x patch x patch x ninstance) X 2
            loss_instance = bs_loss(onehot_instance_gt, onehot_instance_pred) #size 2
            loss_instance = tf.reduce_mean(loss_instance)
        else:
            loss_instance = tf.reduce_mean(loss_foreback)

        loss = class_weights[0, 0] * loss_denoise + class_weights[0, 1] * loss_instance +\
               class_weights[0, 3] * loss_foreback[0] + class_weights[0, 2] * loss_foreback[1] #+ 0.001*reg_laplace

        return loss


    def rec(y_true, y_pred):

        ## Get gt inputs
        rec_gt, n2v_mask, foreback_gt, instance_gt = get_gt(y_true)

        ## Get predictions
        if ninstance > 0:
            foreback_pred, instance_pred = get_estimation(y_pred)
        else:
            foreback_pred = get_estimation(y_pred)

        ## Foreground- background reconstruction loss
        foreback_pred = tf.reshape(foreback_pred,
                                   [-1, patch_size[0], patch_size[1], nchannels, nfore + nback, ncomponents])
        foreback_mask = tf.nn.softmax(foreback_pred[..., 0],
                                      axis=-1)  # nbatch x patch x patch x nchannels x (nfore + nback)
        y_mean, y_log_std = mean_std(
            foreback_pred[..., 1:])  # nbatch x 1 x 1 x nchannels x (nfore + nback)  [both same size or 0.0]


        # print(rec_gt.shape, y_mean.shape)
        loss_denoise = rec_loss(tf.expand_dims(rec_gt, axis=-1), y_mean, y_log_std) * foreback_mask  # nbatch x w x h x nchannels x (nfore + nback)
        loss_denoise = tf.reduce_sum(loss_denoise, -1) # nbatch x w x h x nchannels
        loss_denoise = tf.reduce_sum(loss_denoise * n2v_mask) / tf.math.maximum(tf.reduce_sum(n2v_mask),1)

        return loss_denoise

    def entropy(y_true, y_pred):
        ## Get predictions
        if ninstance > 0:
            foreback_pred, instance_pred = get_estimation(y_pred)
        else:
            foreback_pred = get_estimation(y_pred)

        ## Foreground- background
        foreback_pred = tf.reshape(foreback_pred, [-1, patch_size[0], patch_size[1], nchannels, nfore + nback, ncomponents])
        foreback_mask = tf.nn.softmax(foreback_pred[..., 0], axis=-1)  # nbatch x patch x patch x nchannels x (nfore + nback)
        onehot_pred = tf.reshape(tf.stack([tf.expand_dims(tf.reduce_sum(foreback_mask[..., 0:nfore], -1), axis=-1),
                                           tf.expand_dims(tf.reduce_sum(foreback_mask[..., nfore:], -1), axis=-1)],
                                          axis=-1), [-1, 2]) # (nbatch x w x h x nchannels) X 2

        loss_entropy = tf.reduce_sum(-1*tf.log(tf.math.maximum(onehot_pred,1e-20))*onehot_pred,-1)
        loss_entropy = tf.reduce_mean(loss_entropy)  # reduction of entropy maps

        return loss_entropy

    def seg_error(y_true, y_pred):
        ## TODO!!: This should be rename to fgbg error

        _, _, foreback_gt, _ = get_gt(y_true)
        if ninstance > 0:
            foreback_pred, _= get_estimation(y_pred)
        else:
            foreback_pred = get_estimation(y_pred)

        ## Foreground- background reconstruction
        foreback_pred = tf.reshape(foreback_pred, [-1, patch_size[0], patch_size[1], nchannels, nfore + nback, ncomponents])
        foreback_mask = tf.nn.softmax(foreback_pred[..., 0], axis=-1)  # nbatch x patch x patch x nchannels x (nfore + nback)

        #annotations foreback loss
        onehot_gt = tf.reshape(foreback_gt, [-1, 2])  # (nbatchxwxhxnchannels) x 2
        onehot_pred = tf.reshape(tf.stack([tf.expand_dims(tf.reduce_sum(foreback_mask[..., 0:nfore], -1), axis=-1),
                                           tf.expand_dims(tf.reduce_sum(foreback_mask[..., nfore:], -1), axis=-1)],
                                          axis=-1), [-1, 2])
        loss_foreback = bs_loss(onehot_gt, onehot_pred) #size 2

        return loss_foreback

    def multich_seg_error(y_true, y_pred):
        ## TODO!!: This should be rename to instance error
        if ninstance>0:
            _, _, _, instance_gt = get_gt(y_true)
            _, instance_pred = get_estimation(y_pred)

            #instance loss
            instance_pred = tf.reshape(instance_pred, [-1, patch_size[0], patch_size[1], ninstance, nforei+nbacki])
            instance_pred = tf.nn.softmax(instance_pred, axis=-1)  # nbatch x patch x patch x nchannels x ninstance x 2

            onehot_instance_gt = tf.reshape(instance_gt, [-1, 2])
            # onehot_instance_pred = tf.reshape(instance_pred, [-1, 2])
            onehot_instance_pred = tf.reshape(tf.stack([(tf.reduce_sum(instance_pred[..., 0:nforei], -1)),
                                                        (tf.reduce_sum(instance_pred[..., nforei:], -1))],
                                                       axis=-1), [-1, 2])  # (nbatch x patch x patch x ninstance) X 2
            loss_instance = bs_loss(onehot_instance_gt, onehot_instance_pred)  # size 2

            loss_instance = tf.reduce_mean(loss_instance)
        else:
            loss_instance = tf.convert_to_tensor(np.array([0]))

        return loss_instance



    if type_loss is 'full':
        return demix
    if type_loss is 'rec':
        return rec
    if type_loss is 'entropy':
        return entropy
    if type_loss is 'multich_seg_error':
        return multich_seg_error
        # if ninstance>0:
            # return multich_seg_error
        # else:
            # return seg_error
    if type_loss is 'seg_error':
        return seg_error