from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict, backend_channels_last
import numpy as np
import keras.backend as K
import argparse
from distutils.util import strtobool
from skimage import morphology
import sys

# class BioSegConfig(N2VConfig):
class BioSegConfig(argparse.Namespace):


    def __init__(self, X,**kwargs):

        if  X.size != 0:

            assert len(X.shape) == 4 or len(X.shape) == 5, "Only 'SZYXC' or 'SYXC' as dimensions is supported."

            n_dim = len(X.shape) - 2

            if n_dim == 2:
                axes = 'SYXC'
            elif n_dim == 3:
                axes = 'SZYXC'

            # parse and check axes
            axes = axes_check_and_normalize(axes)
            ax = axes_dict(axes)
            ax = {a: (ax[a] is not None) for a in ax}

            (ax['X'] and ax['Y']) or _raise(ValueError('lateral axes X and Y must be present.'))
            not (ax['Z'] and ax['T']) or _raise(ValueError('using Z and T axes together not supported.'))

            axes.startswith('S') or (not ax['S']) or _raise(ValueError('sample axis S must be first.'))
            axes = axes.replace('S', '')  # remove sample axis if it exists

            if backend_channels_last():
                if ax['C']:
                    axes[-1] == 'C' or _raise(ValueError('channel axis must be last for backend (%s).' % K.backend()))
                else:
                    axes += 'C'
            else:
                if ax['C']:
                    axes[0] == 'C' or _raise(ValueError('channel axis must be first for backend (%s).' % K.backend()))
                else:
                    axes = 'C' + axes

            means, stds = [], []
            for i in range(X.shape[-1]):
                means.append(np.mean(X[..., i]))
                stds.append(np.std(X[..., i]))

            # normalization parameters
            self.means = [str(el) for el in means]
            self.stds = [str(el) for el in stds]
            # directly set by parameters
            self.n_dim = n_dim
            self.axes = axes
            # fixed parameters
            if 'C' in axes:
                self.n_channel_in = X.shape[-1]
            else:
                self.n_channel_in = 1
            self.train_loss = 'demix'

            # default config (can be overwritten by kwargs below)

            self.unet_n_depth = 2
            self.unet_kern_size = 3
            self.unet_n_first = 64
            self.unet_last_activation = 'linear'
            self.probabilistic = False
            self.unet_residual = False
            if backend_channels_last():
                self.unet_input_shape = self.n_dim * (None,) + (self.n_channel_in,)
            else:
                self.unet_input_shape = (self.n_channel_in,) + self.n_dim * (None,)

            # fixed parameters
            self.train_epochs = 200
            self.train_steps_per_epoch = 50
            self.train_learning_rate = 0.0004
            self.train_batch_size = 64
            self.train_tensorboard = False
            self.train_checkpoint = 'weights_best.h5'
            self.train_checkpoint_last  = 'weights_last.h5'
            self.train_checkpoint_epoch = 'weights_now.h5'
            self.train_reduce_lr = {'monitor': 'val_loss', 'factor': 0.5, 'patience': 10}
            self.batch_norm = False
            self.n2v_perc_pix = 1.5
            self.n2v_patch_shape = (128, 128) if self.n_dim == 2 else (64, 64, 64)
            self.n2v_manipulator = 'uniform_withCP'
            self.n2v_neighborhood_radius = 5

            self.single_net_per_channel = False


            self.channel_denoised = False
            self.multi_objective = True
            self.normalizer = 'none'
            self.weights_objectives = [0.1, 0, 0.45, 0.45]
            self.distributions = 'gauss'
            self.n2v_leave_center = False
            self.scale_aug = False
            self.structN2Vmask = None

            self.n_back_modes = 2
            self.n_fore_modes = 2
            self.n_instance_seg = 0
            self.n_back_i_modes = 1
            self.n_fore_i_modes = 1

            self.fit_std = False
            self.fit_mean = True



            # self.n_channel_out = (self.n_back_modes + self.n_fore_modes) * self.n_channel_in * self.n_components +\
            #                      self.n_instance_seg * (self.n_back_i_modes+self.n_fore_i_modes)

        try:
            kwargs['probabilistic'] = False
        except:
            pass
        # disallow setting 'unet_residual' manually
        try:
            kwargs['unet_residual'] = False
        except:
            pass

        # print('KWARGS')
        for k in kwargs:

            # print(k,  kwargs[k])
            setattr(self, k, kwargs[k])
        self.n_components = 3 if (self.fit_std & self.fit_mean) else 2
        self.n_channel_out = (self.n_back_modes + self.n_fore_modes) * self.n_channel_in * self.n_components + \
                             self.n_instance_seg * (self.n_back_i_modes + self.n_fore_i_modes)


        # try:
        #     kwargs['n_channel_out'] = (self.n_back_modes + self.n_fore_modes) * self.n_channel_in * self.n_components +\
        #                               self.n_instance_seg * (self.n_back_i_modes+self.n_fore_i_modes)
        # except:
        #     pass

    def is_valid(self, return_invalid=False):

        ## Todo! Update this properly
        ok = {}
        ok['train_loss'] = (self.train_loss in ['demix'])

        # print(self.channel_denoised)
        ok['channel_denoised'] = isinstance(self.channel_denoised,bool)
        ok['multi_objective'] = isinstance(self.multi_objective,bool)
        ok['distributions'] = (self.distributions in ['gauss','laplace'])
        ok['n2v_leave_center'] = isinstance(self.n2v_leave_center,bool)
        ok['scale_aug'] = isinstance(self.scale_aug, bool)

        if return_invalid:
            return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        else:
            return all(ok.values())

    def update_parameters(self, allow_new=True, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])


def make_defconfig():
    # Make config
    config = argparse.ArgumentParser()

    # Basic configuration
    config.add_argument('--channel_denoised', action='store', default=False, type=lambda x: bool(strtobool(x)),
                        help='boolean: channel_denoised?')
    config.add_argument('--multi_objective', action='store', default=True, type=lambda x: bool(strtobool(x)),
                        help='boolean: multi_objective?')
    config.add_argument('--train_batch_size', action='store', default=128, type=int,
                        help='train_batch_size')
    config.add_argument('--n_channel_out', action='store', default=9, type=int,
                        help='n_channel_out')
    config.add_argument('--epochs', action='store', default=100, type=int,
                        help='train_epochs')
    config.add_argument('--train_loss', action='store', default='demix', type=str, help='train_loss')
    config.add_argument('--normalizer', action='store', default='none', type=str, help='normalizer')

    config.add_argument('--lr', action='store', default=4e-4, type=float, help='learning rate')
    config.add_argument('--seed', action='store', default=1, type=int, help='randomizer seed')
    config.add_argument('--batchnorm', action='store', default=False, type=lambda x: bool(strtobool(x)),
                        help='boolean: batchnorm')

    config.add_argument('--n2v_man', action='store', default='uniform_withCP', type=str, help='n2v_manipulator')
    config.add_argument('--n2v_radius', action='store', default=5, type=int, help='n2v_radius sampler center')
    config.add_argument('--n2v_struct_radius', action='store', default=0, type=int,
                        help='n2v_struct_radius sampler center')

    config.add_argument('--unet_n_depth', action='store', default=4, type=int, help='UNET depth')

    config.add_argument('--leave_center', action='store', default=False, type=lambda x: bool(strtobool(x)),
                        help='boolean: local sampled value of imputations center using manipulator?')
    config.add_argument('--scale_aug', action='store', default=True, type=lambda x: bool(strtobool(x)),
                        help='boolean: scale augmentation?')

    ## Savings
    config.add_argument('--model_name', action='store', default='bioseg', type=str, help='model_name')
    config.add_argument('--basedir', action='store', default='models/', type=str,
                        help='basedir for internal model save')
    config.add_argument('--load_tf_file', action='store', default='', type=str, help='load_tf_file')

    config.add_argument('--weights', action='store',
                        default='1_3_3_3',
                        type=str, help='weight objectives : int_int_int_int')
    config.add_argument('--bf_dist', action='store',
                        default="gauss",
                        type=str, help='background and foreground distribution: gauss , laplace supported')

    argv = sys.argv

    args, unknown = config.parse_known_args(argv)

    config_dict = args.__dict__
    if config_dict['n2v_struct_radius'] == 0:
        config_dict['structN2Vmask'] = None
    else:
        structN2Vmask = morphology.disk(config_dict['n2v_struct_radius'])
        config_dict['structN2Vmask'] = structN2Vmask.tolist()

    config_dict['steps_per_epoch'] = 50
    config_dict['n2v_patch_shape'] = (128,128)
    config_dict['backfore_distribution'] = 'gauss'
    config_dict['n_back_modes'] = 2
    config_dict['n_fore_modes'] = 1
    config_dict['n_back_i_modes'] = 2
    config_dict['n_fore_i_modes'] = 1
    config_dict['n_instance_seg'] = 1
    config_dict['fit_std'] = True
    config_dict['fit_mean'] = True
    config_dict['unet_n_first'] = 32

    return config_dict