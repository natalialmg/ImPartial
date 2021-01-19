from keras.callbacks import TerminateOnNaN, Callback, TensorBoard, ReduceLROnPlateau
from csbdeep.utils.tf import _raise
from csbdeep.models import CARE
from csbdeep.utils import axes_check_and_normalize, axes_dict, move_image_axes
from csbdeep.data import PadAndCropResizer
from n2v.utils.n2v_utils import *

from six import string_types
from csbdeep.utils.six import Path, FileNotFoundError
from csbdeep.internals import nets, predict
from csbdeep.utils import normalize
import datetime

from BioSeg_dataloaders import BioSeg_DataWrapper, manipulate_val_data, subpatch_2D
from losses import *
from BioSeg_config import *
from utils import *

import os
import numpy as np
import keras
import warnings

import tensorflow as tf
from csbdeep.utils import load_json

Optimizer = keras.optimizers.Optimizer

class BioSeg_standard(CARE):

    def __init__(self, config, name=None, basedir='.'):
        """See class docstring."""
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))
        (not (config is None and basedir is None)) or _raise(
            ValueError("No config provided and cannot be loaded from disk since basedir=None."))

        name is None or (isinstance(name, string_types) and len(name) > 0) or _raise(
            ValueError("No valid name: '%s'" % str(name)))
        basedir is None or isinstance(basedir, (string_types, Path)) or _raise(
            ValueError("No valid basedir: '%s'" % str(basedir)))
        self.config = config
        self.name = name if name is not None else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.basedir = Path(basedir) if basedir is not None else None
        if config is not None:
            # config was provided -> update before it is saved to disk
            self._update_and_check_config()
        self._set_logdir()
        if config is None:
            # config was loaded from disk -> update it after loading
            self._update_and_check_config()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()

    def _build(self):
        return self._build_unet(
            n_dim=self.config.n_dim,
            n_channel_out=self.config.n_channel_out,
            residual=self.config.unet_residual,
            n_depth=self.config.unet_n_depth,
            kern_size=self.config.unet_kern_size,
            n_first=self.config.unet_n_first,
            last_activation=self.config.unet_last_activation,
            batch_norm=self.config.batch_norm
        )(self.config.unet_input_shape)

    def _build_unet(self, n_dim=2, n_depth=4, kern_size=3, n_first=32, n_channel_out=1, residual=True,
                    last_activation='linear', batch_norm=True, single_net_per_channel=False):
        """Construct a common CARE neural net based on U-Net [1]_ and residual learning [2]_ to be used for image restoration/enhancement.
           Parameters
           ----------
           n_dim : int
               number of image dimensions (2 or 3)
           n_depth : int
               number of resolution levels of U-Net architecture
           kern_size : int
               size of convolution filter in all image dimensions
           n_first : int
               number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
           n_channel_out : int
               number of channels of the predicted output image
           residual : bool
               if True, model will internally predict the residual w.r.t. the input (typically better)
               requires number of input and output image channels to be equal
           last_activation : str
               name of activation function for the final output layer
           batch_norm : bool
               Use batch normalization during training
           Returns
           -------
           function
               Function to construct the network, which takes as argument the shape of the input image
           Example
           -------
           >>> model = common_unet(2, 2, 3, 32, 1, True, 'linear', True)(input_shape)
           References
           ----------
           .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
           .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
           """

        def _build_this(input_shape):
            # if single_net_per_channel:
            #     return build_single_unet_per_channel(input_shape, last_activation, n_depth, n_first,
            #                                          (kern_size,) * n_dim,
            #                                          pool_size=(2,) * n_dim, residual=residual, prob_out=False,
            #                                          batch_norm=batch_norm)
            # else:
            return nets.custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,) * n_dim,
                                        pool_size=(2,) * n_dim, n_channel_out=n_channel_out, residual=residual,
                                        prob_out=False, batch_norm=batch_norm)

        return _build_this

        # @property
        # def _config_class(self):
        #     return N2VConfig

    def prepare_model_bioseg(self, model, optimizer, loss):

        weights = self.config.weights_objectives
        backfore_distribution = self.config.distributions

        if loss in ['demix']:
            # print('masked')
            # print(self.config.n_instance_seg)
            isinstance(optimizer, Optimizer) or _raise(ValueError())
            loss_standard = eval('loss_{}(relative_weights = {},type_loss = "full",'
                                 ' type_rec = "{}",patch_size = "{}",nchannels="{}" ,'
                                 'nback ="{}", nfore="{}",'
                                 'nbacki ="{}", nforei="{}", ninstance ="{}",std ="{}", mean ="{}")'.format(loss, weights,backfore_distribution,
                                                                                              self.config.n2v_patch_shape,self.config.n_channel_in,self.config.n_back_modes,
                                                                                    self.config.n_fore_modes,self.config.n_back_i_modes,self.config.n_fore_i_modes, self.config.n_instance_seg,
                                                                                                          self.config.fit_std,self.config.fit_mean))

            _metrics = [] #rec
            _metrics.append(eval('loss_{}(relative_weights = {},type_loss = "rec",'
                                 ' type_rec = "{}",patch_size = "{}",nchannels="{}" ,'
                                 'nback ="{}", nfore="{}",'
                                 'nbacki ="{}", nforei="{}", ninstance ="{}",std ="{}", mean ="{}")'.format(loss, weights,backfore_distribution,
                                                                                              self.config.n2v_patch_shape,self.config.n_channel_in,self.config.n_back_modes,
                                                                                    self.config.n_fore_modes,self.config.n_back_i_modes,self.config.n_fore_i_modes, self.config.n_instance_seg,
                                                                                                          self.config.fit_std,self.config.fit_mean)))
            _metrics.append(eval('loss_{}(relative_weights = {},type_loss = "entropy",'
                                 ' type_rec = "{}",patch_size = "{}",nchannels="{}" ,'
                                 'nback ="{}", nfore="{}",'
                                 'nbacki ="{}", nforei="{}", ninstance ="{}",std ="{}", mean ="{}")'.format(loss, weights,backfore_distribution,
                                                                                              self.config.n2v_patch_shape,self.config.n_channel_in,self.config.n_back_modes,
                                                                                    self.config.n_fore_modes,self.config.n_back_i_modes,self.config.n_fore_i_modes, self.config.n_instance_seg,
                                                                                                          self.config.fit_std,self.config.fit_mean)))
            if self.config.multi_objective:
                # print('MULTI')
                _metrics.append(eval('loss_{}(relative_weights = {},type_loss = "multich_seg_error",'
                                 ' type_rec = "{}",patch_size = "{}",nchannels="{}" ,'
                                 'nback ="{}", nfore="{}",'
                                 'nbacki ="{}", nforei="{}", ninstance ="{}",std ="{}", mean ="{}")'.format(loss, weights,backfore_distribution,
                                                                                              self.config.n2v_patch_shape,self.config.n_channel_in,self.config.n_back_modes,
                                                                                    self.config.n_fore_modes,self.config.n_back_i_modes,self.config.n_fore_i_modes, self.config.n_instance_seg,
                                                                                                          self.config.fit_std,self.config.fit_mean)))
                _metrics.append(eval('loss_{}(relative_weights = {},type_loss = "seg_error",'
                                 ' type_rec = "{}",patch_size = "{}",nchannels="{}" ,'
                                 'nback ="{}", nfore="{}",'
                                 'nbacki ="{}", nforei="{}", ninstance ="{}",std ="{}", mean ="{}")'.format(loss, weights,backfore_distribution,
                                                                                              self.config.n2v_patch_shape,self.config.n_channel_in,self.config.n_back_modes,
                                                                                    self.config.n_fore_modes,self.config.n_back_i_modes,self.config.n_fore_i_modes, self.config.n_instance_seg,
                                                                                                          self.config.fit_std,self.config.fit_mean)))

            callbacks = [TerminateOnNaN()]

            # compile model
            model.compile(optimizer=optimizer, loss=loss_standard, metrics=_metrics)

            return callbacks


        else:
            print('!!!!!! #### Error::  Uknown Loss ######')

    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.
        Calls :func:`csbdeep.internals.train.prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.
        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.
        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`csbdeep.internals.train.prepare_model`.
        """

        if optimizer is None:
            Adam = keras.optimizers.adam
            # optimizer = Adam(lr=self.config.train_learning_rate, clipnorm=0.001)
            # optimizer = Adam(lr=self.config.train_learning_rate, clipnorm = 1.0, clipvalue = 0.5)
            optimizer = Adam(lr=self.config.train_learning_rate, clipvalue = 0.5)

        ## Prepare model ##
        self.callbacks = self.prepare_model_bioseg(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.basedir is not None:
            if self.config.train_checkpoint is not None:
                from keras.callbacks import ModelCheckpoint
                self.callbacks.append(
                    ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True,
                                    save_weights_only=True))
                self.callbacks.append(
                    ModelCheckpoint(str(self.logdir / 'weights_now.h5'), save_best_only=False, save_weights_only=True))

            # TODO!!::
            #if self.config.train_tensorboard:
            #    self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3,
            #                                               write_images=True, prob_out=self.config.probabilistic))
                # else:
                #     self.callbacks.append(
                #         TensorBoard(log_dir=str(self.logdir / 'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            from keras.callbacks import ReduceLROnPlateau
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))
        self._model_prepared = True

    def train(self, X,Y, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.
        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.
        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.
        """
        leave_center = self.config.n2v_leave_center
        scale_augmentation = self.config.scale_aug

        ## Resize validation if necessary....
        print((np.sum(np.abs(np.array(validation_data[0][1:-1]) - np.array(self.config.n2v_patch_shape))) != 0))
        if (np.sum(np.abs(np.array(validation_data[0][1:-1]) - np.array(self.config.n2v_patch_shape))) != 0) :
            X_val = subpatch_2D(validation_data[0], np.array(self.config.n2v_patch_shape))
            Y_val = subpatch_2D(validation_data[1], np.array(self.config.n2v_patch_shape))
            validation_data = (X_val,Y_val)




        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        n_train, n_val = len(X), len(validation_data[0])

        ## Warning about validation size
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))

        #axes description
        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
        ax = axes_dict(axes)

        # for a,div_by in zip(axes,self._axes_div_by(axes)):
        #     n = X.shape[ax[a]]
        #     if n % div_by != 0:
        #         raise ValueError(
        #             "training images must be evenly divisible by %d along axis %s"
        #             " (which has incompatible size %d)" % (div_by,a,n)
        #         )


        ## ToDO: what is this??
        div_by = 2 ** self.config.unet_n_depth
        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
        val_num_pix = 1
        train_num_pix = 1
        val_patch_shape = ()
        for a in axes_relevant:
            n = X.shape[ax[a]]
            val_num_pix *= validation_data[0].shape[ax[a]]
            train_num_pix *= X.shape[ax[a]]
            val_patch_shape += tuple([validation_data[0].shape[ax[a]]])
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axes %s"
                    " (axis %s has incompatible size %d)" % (div_by, axes_relevant, a, n)
                )

        # epochs & steps per epochs
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        # if (self.config.train_tensorboard and self.basedir is not None and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks)):
        #     self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
        #                                                log_dir=str(self.logdir/'logs'/'images'),
        #                                                n_images=3, prob_out=self.config.probabilistic))
        #
        # training_data = DataWrapper(X, Y, self.config.train_batch_size,epochs*steps_per_epoch)

        manipulator = eval('pm_{0}({1})'.format(self.config.n2v_manipulator, str(self.config.n2v_neighborhood_radius)))

        if self.config.normalizer is 'std':
            means = np.array([float(mean) for mean in self.config.means], ndmin=len(X.shape), dtype=np.float32)
            stds = np.array([float(std) for std in self.config.stds], ndmin=len(X.shape), dtype=np.float32)

            X = self.__normalize__(X, means, stds)
            validation_X = self.__normalize__(validation_data[0], means, stds)
        else:
            validation_X = validation_data[0]
        # Todo: validation normalization if we have; also pick type of normalization as an option


        #mask (struct to inpaint)
        _mask = np.array(self.config.structN2Vmask) if self.config.structN2Vmask else None
        # print(_mask,self.config.channel_denoised)
        training_data = BioSeg_DataWrapper(X, Y,self.config.train_batch_size, self.config.n2v_perc_pix,
                                        self.config.n2v_patch_shape, manipulator, structN2Vmask=_mask,
                                           chan_denoise=self.config.channel_denoised,
                                           multiple_objectives=self.config.multi_objective,
                                           leave_center=leave_center,scale_augmentation=scale_augmentation)

        # validation_Y is also validation_X plus a concatenated masking channel.
        # To speed things up, we precompute the masking vo the validation data.

        if not self.config.channel_denoised:
            validation_Y = np.concatenate((validation_X, np.zeros(validation_X.shape, dtype=validation_X.dtype)),
                                          axis=axes.index('C'))
        else:
            val_aux = validation_data[1][...,0:X.shape[-1]]
            # print(val_aux.shape)
            # if X.shape[-1] == 1:
            #     val_aux = val_aux[...,np.newaxis]
            validation_Y = np.concatenate((val_aux, np.zeros(val_aux.shape, dtype=validation_X.dtype)),
                                          axis=axes.index('C'))

        # print(validation_Y.shape, validation_X.shape)




        manipulate_val_data(validation_X, validation_Y,
                                      perc_pix=self.config.n2v_perc_pix,
                                      shape=val_patch_shape,
                                      value_manipulation=manipulator,
                                        chan_denoise=self.config.channel_denoised)

        # print(self.config)
        # print(self.config.multi_objective)

        if self.config.multi_objective:
            if (self.config.channel_denoised) and (validation_data[1].shape[-1] > X.shape[-1]): #additional channels
                validation_Y = np.concatenate((validation_Y, validation_data[1][...,X.shape[-1]:]), axis=-1)

            if not self.config.channel_denoised:
                validation_Y = np.concatenate((validation_Y, validation_data[1][...,:]), axis=-1)

        # print(validation_Y.shape, validation_X.shape)
        fit = self.keras_model.fit_generator

        # fit = self.keras_model.fit

        history = fit(training_data, validation_data=(validation_X, validation_Y),
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      callbacks=self.callbacks, verbose=1)


        ## ToDo : what does this save do
        if self.basedir is not None:
            self.keras_model.save_weights(str(self.logdir / 'weights_last.h5'))

            if self.config.train_checkpoint is not None:
                print()
                self._find_and_load_weights(self.config.train_checkpoint)
                try:
                    # remove temporary weights
                    (self.logdir / 'weights_now.h5').unlink()
                except FileNotFoundError:
                    pass


        #self._training_finished()

        return history
        # return validation_Y,validation_X,training_data

    def predict(self, img, axes):
        """
        Apply the network to so far unseen data.
        Parameters
        ----------
        img     : array(floats) of images
        axes    : String
                  Axes of the image ('YX').
        Returns
        -------
        image : array(float)
                The restored image.
        """
        # print('predict!!')
        # print(self.config.normalizer)
        if self.config.normalizer == 'std':
            means = np.array([float(mean) for mean in self.config.means], ndmin=len(img.shape), dtype=np.float32)
            stds = np.array([float(std) for std in self.config.stds], ndmin=len(img.shape), dtype=np.float32)

            if img.dtype != np.float32:
                print('The input image is of type {} and will be casted to float32 for prediction.'.format(img.dtype))
                img = img.astype(np.float32)

            # new_axes = axes
            new_axes = axes
            if 'C' in axes:
                new_axes = axes.replace('C', '') + 'C'
                normalized = self.__normalize__(np.moveaxis(img, axes.index('C'), -1), means, stds)
            else:
                normalized = self.__normalize__(img[..., np.newaxis], means, stds)
                normalized = normalized[..., 0]
        else:

            if img.dtype != np.float32:
                print('The input image is of type {} and will be casted to float32 for prediction.'.format(img.dtype))
                img = img.astype(np.float32)

            # new_axes = axes
            if 'C' not in axes:
                img = img[..., np.newaxis]
            if 'S' not in axes:
                img = img[np.newaxis,...]

            normalized = img+0

        pred_full = self.keras_model.predict(normalized)

        if pred_full.shape[0] == 1:
            pred_full = pred_full[0,...]


        # if 'C' in axes:
        #     pred = np.moveaxis(pred, -1, axes.index('C'))

        return pred_full

    #ToDo: add Normalize percentual
    def __normalize__(self, data, means, stds):
        return (data - means) / stds

    def __denormalize__(self, data, means, stds):
        return (data * stds) + means

def set_bioseg_model(basedir, model_name, BioConfig, load_tf_file = '', load_model=False, seed = 0):
    mkdir(basedir)

    print('------------------  BioSeg setup  -------------------------')
    print('basedir/model name : ', basedir+model_name)
    print('model file : ',basedir+model_name+'/'+BioConfig.train_checkpoint)
    print()

    if seed > 0:
        tf.set_random_seed(seed)

    if load_model:
        print('** Loading Model Config **')
        print()
        if os.path.exists(basedir+model_name+'/config.json'):

            print('Loading previous config file : ')
            config_dict = load_json(basedir + model_name + '/config.json')
            BioConfig = BioSegConfig(np.array([]), **config_dict)
            print(basedir + model_name + '/config.json')
            if load_tf_file == '': #load default best model in folder
                load_tf_file = basedir+model_name+'/'+BioConfig.train_checkpoint

    print('** Create Model **')
    print()
    print(BioConfig)

    model = BioSeg_standard(BioConfig, model_name, basedir=basedir)
    model.prepare_for_training()

    if os.path.exists(load_tf_file):
        print('Loading model weights : ', load_tf_file)
        model.keras_model.load_weights(load_tf_file)
    else:
        print('Model with random initialization')

    return model

def model_trainer_fc(model):

    def trainer(data):
        X_train = data[0]
        Y_train = data[1]
        X_val = data[2]
        Y_val = data[3]

        history = model.train(X_train, Y_train, (X_val, Y_val))

        history = history.history

        save_data_model = {}
        save_data_model['history'] = history
        save_data_model['basedir'] = model.basedir
        save_data_model['model_folder'] = model.name

        return save_data_model

    return trainer

def model_demix_prediction(model, input_image):
    if len(input_image.shape) == 4:
        y_pred = model.predict(input_image, 'SYXC')
    else:
        if len(input_image.shape) == 3:
            y_pred = model.predict(input_image, 'YXC')
        else:
            y_pred = model.predict((input_image), 'YX')

    if len(y_pred.shape) <= 3:
        y_pred = y_pred[np.newaxis, ...]

    nfore = model.config.n_fore_modes
    nback = model.config.n_back_modes
    nchannels = model.config.n_channel_in
    ncomponents = model.config.n_components
    ninstance = model.config.n_instance_seg
    nfore_instance = model.config.n_fore_i_modes
    nback_instance = model.config.n_back_i_modes

    patch_size = (y_pred.shape[1], y_pred.shape[2])

    Y_est = {}

    ix = 0
    foreback_pred = y_pred[..., ix:ix + (nfore + nback) * nchannels * ncomponents]
    # print(foreback_pred.shape)

    if ninstance>0:
        ix += (nfore + nback) * nchannels * ncomponents
        instance_pred = y_pred[..., ix:ix + ninstance * (nfore_instance+nback_instance)]
    # instance_pred = y_pred[..., ix:]

    ## Foreback mask !!
    foreback_pred = tf.reshape(foreback_pred, [-1, patch_size[0], patch_size[1],
                                               nchannels, nfore + nback, ncomponents])
    foreback_mask = tf.nn.softmax(foreback_pred[..., 0],
                                  axis=-1)  # component 0 is the mixing image, 1 and 2 are mean and std
    foreback_mask = foreback_mask.eval()

    for ch in range(nchannels):
        Y_est['channel_' + str(ch)] = foreback_mask[:, :, :, ch, :]
        Y_est['params_channel_'+ str(ch)] = foreback_pred[:, :, :, ch, :,1:].eval()

    #
    # foreback_mask = np.reshape(foreback_mask, (foreback_mask.shape[0],
    #                                            foreback_mask.shape[1],
    #                                            foreback_mask.shape[2], -1))

    if ninstance>0:
        ## Instance mask !!
        instance_pred = tf.reshape(instance_pred, [-1, patch_size[0], patch_size[1],
                                                   ninstance, nfore_instance+nback_instance])
        instance_mask = tf.nn.softmax(instance_pred, axis=-1)
        instance_mask = instance_mask.eval()
        for ni in range(ninstance):
            Y_est['instance_' + str(ni)] = instance_mask[:, :, :, ni, :]

    # instance_mask = np.reshape(instance_mask, (instance_mask.shape[0],
    #                                            instance_mask.shape[1],
    #                                            instance_mask.shape[2], -1))

    # print('foreback/Instance mask shape : ',foreback_mask.shape,instance_mask.shape)

    return Y_est

def model_evaluation_fc(model, perc_normalization=True, pmin=1, pmax=99.8):

    perc_normalization = perc_normalization
    pmin = pmin
    pmax = pmax

    def model_evaluation(raw_image):
        if perc_normalization:
            raw_image_in = normalize(raw_image, pmin=pmin, pmax=pmax, clip=False)
        else:
            raw_image_in = raw_image + 0

        ### Evaluate Model ###
        Y_est = model_demix_prediction(model, raw_image_in)

        return Y_est

    return model_evaluation



