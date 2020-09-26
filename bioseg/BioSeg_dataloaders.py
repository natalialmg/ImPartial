from keras.utils import Sequence
from n2v.utils.n2v_utils import *
from tqdm import tqdm
import numpy as np


class BioSeg_DataWrapper(Sequence):
    def __init__(self, X, seg_Y, batch_size, perc_pix, shape, value_manipulation,
                 structN2Vmask=None,chan_denoise = False,
                 multiple_objectives = False, leave_center = True,
                 scale_augmentation = False):
        assert X.shape[0] == seg_Y.shape[0]
        self.X, self.seg_Y = X,  seg_Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))
        self.shape = shape
        self.value_manipulation = value_manipulation
        self.range = np.array(self.X.shape[1:-1]) - np.array(self.shape)
        self.dims = len(shape)
        self.n_chan = X.shape[-1]
        self.structN2Vmask = structN2Vmask
        self.leave_center = leave_center
        self.scale_augmentation = scale_augmentation

        if self.structN2Vmask is not None:
            print("StructN2V Mask is: ", self.structN2Vmask)

        num_pix = int(np.product(shape) / 100.0 * perc_pix)
        assert num_pix >= 1, "Number of blind-spot pixels is below one. At least {}% of pixels should be replaced.".format(
            100.0 / np.product(shape))
        print("{} blind-spots will be generated per training patch of size {}.".format(num_pix, shape))

        if self.dims == 2:
            self.patch_sampler = self.__subpatch_sampling2D__
            self.box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)
            self.get_stratified_coords = self.__get_stratified_coords2D__
            self.rand_float = self.__rand_float_coords2D__(self.box_size)
        elif self.dims == 3:
            self.patch_sampler = self.__subpatch_sampling3D__
            self.box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)
            self.get_stratified_coords = self.__get_stratified_coords3D__
            self.rand_float = self.__rand_float_coords3D__(self.box_size)
        else:
            raise Exception('Dimensionality not supported.')

        self.chan_denoise = chan_denoise #Assume that if true first self.n_chan of Y are denoised channels
        self.multiple_objectives = multiple_objectives
        self.n_chan_Y = seg_Y.shape[-1]


        self.X_Batches = np.zeros((self.X.shape[0], *self.shape, self.n_chan), dtype=np.float32)
        self.Y_n2vBatches = np.zeros((self.X.shape[0], *self.shape, 2 * self.n_chan), dtype=np.float32)
        self.Y_segBatches = np.zeros((self.seg_Y.shape[0], *self.shape, self.n_chan_Y), dtype=np.float32)

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))
        self.X_Batches *= 0
        self.Y_n2vBatches *= 0
        self.Y_segBatches *= 0

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.perm[idx]

        ## SAMPLE PATCHES GIVEN A SET OF RANDOM INDICES ##
        self.patch_sampler(self.X, self.X_Batches, self.seg_Y, self.Y_segBatches,
                           indices=idx, range=self.range, shape=self.shape)

        for c in range(self.n_chan):
            for j in idx:

                if self.scale_augmentation:
                    scale_aug = np.random.random(1)*(2-0.5)+0.5
                else:
                    scale_aug = 1

                ## GET COORDINATES OF PIXELS TO REPLACE WITH NOISE ##
                coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                    shape=self.shape)

                indexing = (j,) + coords + (c,)
                indexing_mask = (j,) + coords + (c + self.n_chan,)

                # print('denoise channel:: ',self.chan_denoise)
                y_val = self.X_Batches[indexing]*scale_aug if not self.chan_denoise else self.Y_segBatches[indexing]*scale_aug

                ## SAMPLE VALUE OF PIXELS TO REPLACE
                x_val = self.value_manipulation(self.X_Batches[j, ..., c]*scale_aug, coords, self.dims)



                self.Y_n2vBatches[indexing] = y_val
                self.Y_n2vBatches[indexing_mask] = 1
                self.X_Batches[indexing] = x_val

                #inpainting structure
                if self.structN2Vmask is not None:
                    self.apply_structN2Vmask(self.X_Batches[j, ..., c], coords,
                                             self.structN2Vmask, leave_center = self.leave_center)

        if self.multiple_objectives:
            Y_seg_out = self.Y_segBatches[idx] if not self.chan_denoise else self.Y_segBatches[idx,...,self.n_chan:]
            Y_seg_out = np.concatenate((self.Y_n2vBatches[idx], Y_seg_out), axis=-1)
        else:
            Y_seg_out = self.Y_n2vBatches[idx]
        return self.X_Batches[idx], Y_seg_out


    def apply_structN2Vmask(self, patch, coords, mask, leave_center = True):
        """
        each point in coords corresponds to the center of the mask.
        then for point in the mask with value=1 we assign a random value
        """
        coords = np.array(coords).astype(np.int)
        ndim = mask.ndim
        center = np.array(mask.shape)//2
        ## leave the center value alone
        if leave_center:
            # print('leave center true')
            mask[tuple(center.T)] = 0
        # else:
            # print('leave center false')
        ## displacements from center
        dx = np.indices(mask.shape)[:,mask==1] - center[:,None]
        ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
        mix = (dx.T[...,None] + coords[None])
        mix = mix.transpose([1,0,2]).reshape([ndim,-1]).T
        ## stay within patch boundary
        mix = mix.clip(min=np.zeros(ndim),max=np.array(patch.shape)-1).astype(np.uint)
        ## replace neighbouring pixels with random values from flat dist
        ## Todo: This selection of sampling seems quite arbitrary...
        patch[tuple(mix.T)] = np.random.rand(mix.shape[0])*4 - 2


    @staticmethod
    def __subpatch_sampling2D__(X, X_Batches, Y_seg, Y_segBatches, indices, range, shape):
        for j in indices:
            y_start = np.random.randint(0, range[0] + 1)
            x_start = np.random.randint(0, range[1] + 1)
            X_Batches[j] = np.copy(X[j, y_start:y_start + shape[0], x_start:x_start + shape[1]])
            Y_segBatches[j] = np.copy(Y_seg[j, y_start:y_start + shape[0], x_start:x_start + shape[1]])

    @staticmethod
    def __subpatch_sampling3D__(X, X_Batches, Y_seg, Y_segBatches, indices, range, shape):
            for j in indices:
                z_start = np.random.randint(0, range[0] + 1)
                y_start = np.random.randint(0, range[1] + 1)
                x_start = np.random.randint(0, range[2] + 1)
                X_Batches[j] = np.copy(
                    X[j, z_start:z_start + shape[0], y_start:y_start + shape[1], x_start:x_start + shape[2]])
                Y_segBatches[j] = np.copy(
                    Y_seg[j, z_start:z_start + shape[0], y_start:y_start + shape[1], x_start:x_start + shape[2]])


    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        box_count_y = int(np.ceil(shape[0] / box_size))
        box_count_x = int(np.ceil(shape[1] / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    y_coords.append(y)
                    x_coords.append(x)
        return (y_coords, x_coords)

    @staticmethod
    def __get_stratified_coords3D__(coord_gen, box_size, shape):
        box_count_z = int(np.ceil(shape[0] / box_size))
        box_count_y = int(np.ceil(shape[1] / box_size))
        box_count_x = int(np.ceil(shape[2] / box_size))
        x_coords = []
        y_coords = []
        z_coords = []
        for i in range(box_count_z):
            for j in range(box_count_y):
                for k in range(box_count_x):
                    z, y, x = next(coord_gen)
                    z = int(i * box_size + z)
                    y = int(j * box_size + y)
                    x = int(k * box_size + x)
                    if (z < shape[0] and y < shape[1] and x < shape[2]):
                        z_coords.append(z)
                        y_coords.append(y)
                        x_coords.append(x)
        return (z_coords, y_coords, x_coords)

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

    @staticmethod
    def __rand_float_coords3D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)




## ToDO:!! Val data manipulator does not perform struct removal
def manipulate_val_data(X_val, Y_val,value_manipulation, perc_pix=0.198, shape=(64, 64),chan_denoise = False):
    dims = len(shape)
    if dims == 2:
        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        get_stratified_coords = BioSeg_DataWrapper.__get_stratified_coords2D__
        rand_float = BioSeg_DataWrapper.__rand_float_coords2D__(box_size)
    elif dims == 3:
        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        get_stratified_coords = BioSeg_DataWrapper.__get_stratified_coords3D__
        rand_float = BioSeg_DataWrapper.__rand_float_coords3D__(box_size)

    n_chan = X_val.shape[-1]

    if not chan_denoise:
        Y_val *= 0
    else:
        Y_val[...,n_chan:] *= 0
    for j in tqdm(range(X_val.shape[0]), desc='Preparing validation data: '):
        coords = get_stratified_coords(rand_float, box_size=box_size,
                                            shape=np.array(X_val.shape)[1:-1])
        for c in range(n_chan):
            indexing = (j,) + coords + (c,)
            indexing_mask = (j,) + coords + (c + n_chan,)

            y_val = X_val[indexing] if not chan_denoise else Y_val[indexing]
            x_val = value_manipulation(X_val[j, ..., c], coords, dims)

            Y_val[indexing_mask] = 1
            Y_val[indexing] = y_val

            X_val[indexing] = x_val