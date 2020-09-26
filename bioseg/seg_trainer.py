from preprocessing import *
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, jaccard_score
import pandas as pd
import numpy as np
# from BioSeg_labelgenerator import *
from preprocessing import get_XY_patches_from_files_generic,balance_augment_data
from csbdeep.utils import normalize
import pickle
import re
from skimage import io, morphology
import matplotlib.pyplot as plt
from utils import mkdir

class SegmentationTrainer:
    def __init__(self,load_dic=None, model_input_dic=None, save_dir=None,
                 iteration=0,train_data_path = None,eval_data_path = None):


        if load_dic is None:
            self.iteration = iteration
            self.save_dir = save_dir
            mkdir(self.save_dir)
            self.train_data_path = train_data_path
            self.eval_data_path = eval_data_path
            self.model_input_dic = model_input_dic
            self.model_prefix = model_input_dic['model_name']
            self.basedir_model = model_input_dic['basedir']
            self.history = []
            self.trained_models_dir = []
            self.eval_files = []
        else:
            self.load_params(load_dic)

    def load_params(self, load_dic):

        self.iteration = load_dic['iteration']
        self.save_dir = load_dic['save_dir']
        self.train_data_path = load_dic['train_data_path']
        self.eval_data_path = load_dic['eval_data_path']
        self.model_input_dic = load_dic['model_input_dic']
        self.model_prefix = load_dic['model_prefix']
        self.basedir_model = load_dic['basedir_model']
        self.history = load_dic['history']
        self.trained_models_dir = load_dic['trained_models_dir']
        self.eval_files = load_dic['eval_files']

    def get_params(self):

        load_dic = {}
        load_dic['iteration'] = self.iteration
        load_dic['save_dir'] = self.save_dir
        load_dic['train_data_path'] = self.train_data_path
        load_dic['eval_data_path'] = self.eval_data_path
        load_dic['model_input_dic'] = self.model_input_dic
        load_dic['model_prefix'] = self.model_prefix
        load_dic['basedir_model'] = self.basedir_model
        load_dic['history'] = self.history
        load_dic['trained_models_dir'] = self.trained_models_dir
        load_dic['eval_files'] = self.eval_files

        return load_dic

    def get_new_dataset(self, tag_X_dir='input_dir', tag_Y_dir='label_dir',
                        tag_X_file='input_file',tag_Y_file = 'labels_file', balance=False,
                        npatches_per_image=50, p_label=0.4, save_file = 'dataset',
                        patch_size = (128, 128),plots=False, pnormalization = True):

        pd_files = pd.read_csv(self.train_data_path)
        for tag in ['train', 'val']:
            pd_filter = pd_files.loc[pd_files.group == tag]

            print( 'Scribbles tag :' ,tag_Y_file)

            X, Y = get_XY_patches_from_files_generic(pd_filter, tag_X_dir=tag_X_dir,
                                                     tag_Y_dir=tag_Y_dir,
                                                     tag_X_file=tag_X_file,
                                                     tag_Y_file=tag_Y_file,
                                                     n_patches_per_image=npatches_per_image,
                                                     p_label=p_label,
                                                     patch_size=patch_size,
                                                     pnormalization = pnormalization)

            print(X.shape, Y.shape)
            if tag == 'train':
                X_train, Y_train = balance_augment_data(X, Y, augment=True, balance=balance)
            if tag == 'val':
                X_val, Y_val = balance_augment_data(X, Y, augment=False, balance=balance)

        print()
        print('### SUMMARY DATA ###')
        print('training')
        print(X_train.shape, Y_train.shape)
        print('fraction train patches with annotations : ',np.mean(np.sum(Y_train, axis=(1, 2, 3))>0))
        print('Validation')
        print(X_val.shape, Y_val.shape)
        print('fraction val patches with annotations : ',np.mean(np.sum(Y_val, axis=(1, 2, 3)) > 0))

        data_save_file = self.save_dir + save_file +'.npz'
        np.savez(data_save_file, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)
        print('saving dataset :: ', data_save_file)

        #####################################plot##################################

        if plots:
            print()
            print('#### Patches Instance Scribbles ###')
            plt.figure(figsize=(20, 20))
            ix_plot = np.where(np.sum(Y_train, axis=(1, 2, 3))>0)[0]
            np.random.shuffle(ix_plot)

            for i in np.arange(5):
                plt.subplot(4, 5, i + 1)
                plt.imshow(normalize(np.sum(X_train[ix_plot[i], :, :, :],axis = -1),pmin=1,pmax=99.8,clip=True))


            ## Instance Scribbles ##
            for i in np.arange(5):
                plt.subplot(4, 5, i + 6)
                aux = np.zeros([Y_train.shape[1],Y_train.shape[2],3])
                aux[...,1] = normalize(np.sum(X_train[ix_plot[i], :, :, :],axis = -1),pmin=1,pmax=99.8,clip=True)*0.4
                aux[...,0] = Y_train[ix_plot[i], :, :, -1]
                aux[...,2] = Y_train[ix_plot[i], :, :, -2]
                plt.imshow(aux)
            plt.show()

    def evaluate(self, model_evaluator, path_pandas_eval, suffix='' ,tag_X_dir='input_dir', tag_Y_dir='mask_dir',
                        tag_X_file='input_file',tag_Y_file = 'mask_file', ix_fg = 0, mask_th=0.5, plot=False):

        ###################### Train, Val, Test EVALUATION, file specs on self.pd_data #################

        pd_data = pd.DataFrame()
        pd_files = pd.read_csv(path_pandas_eval)
        pd_data['group'] = pd_files['group'].values
        pd_data['input_dir'] = pd_files[tag_X_dir].values
        pd_data['input_file'] = pd_files[tag_X_file].values
        pd_data['mask_dir'] = pd_files[tag_Y_dir].values
        pd_data['mask_file'] = pd_files[tag_Y_file].values
        pd_data['output_dir'] = [self.save_dir for i in range(len(pd_files))]

        row_estimation = []
        for ix in pd_data.index.values:
            pd_row = pd_data.loc[ix]
            prefix = re.search('.*(?=\.)', pd_row['input_file']).group()
            print(prefix)

            ## raw & gt
            raw_image = io.imread(pd_row['input_dir'] + pd_row['input_file'], as_gray=True)
            npz_gt = np.load(pd_row['mask_dir'] + pd_row['mask_file'])
            labels_gt = npz_gt['foreground']

            ## probability map prediction
            mask_softmax = model_evaluator(raw_image)[0,...]
            # print('Masksoftmax , :', mask_softmax.shape)

            mask_probability = normalize(mask_softmax[..., ix_fg], pmin=1, pmax=99.8, clip=True)
            mask_prediction = np.zeros_like(mask_probability)
            mask_prediction[mask_probability > mask_th] = 1


            mask_gt = np.zeros_like(labels_gt)
            mask_gt[labels_gt > 0] = 1

            np.savez(self.save_dir + prefix + '_estimation_' + str(suffix) + '.npz',
                     outputs=mask_softmax,
                     probability=mask_probability)
            row_estimation.append(prefix + '_estimation_' + str(suffix) + '.npz')

            auc = roc_auc_score(y_true=1 - mask_gt.flatten(), y_score=1 - mask_probability.flatten())
            jacc = jaccard_score(y_true=1 - mask_gt.flatten(), y_pred=1 - mask_prediction.flatten())
            f1 = f1_score(y_true=1 - mask_gt.flatten(), y_pred=1 - mask_prediction.flatten())
            acc = accuracy_score(y_true=1 - mask_gt.flatten(), y_pred=1 - mask_prediction.flatten())

            aux = np.zeros([mask_prediction.shape[0], mask_prediction.shape[1], 3])
            aux[:, :, 1] = mask_prediction * mask_gt * 0.4
            aux[mask_prediction * (1 - mask_gt) > 0, 0] = 1
            aux[(1 - mask_prediction) * mask_gt > 0, 2] = 1

            if plot:
                plt.figure(figsize=(10, 8))
                plt.subplot(2, 2, 1)
                plt.title('raw_image')
                #     plt.imshow(labels_gt)
                plt.imshow(normalize(raw_image, pmin=1, pmax=99.8, clip=True))
                plt.subplot(2, 2, 2)
                plt.title('mask_gt')
                plt.imshow(mask_gt)
                plt.subplot(2, 2, 3)
                plt.title('mask_prediction')
                plt.imshow(mask_probability)
                plt.subplot(2, 2, 4)
                plt.title('3:TP, 2: FN, 1:FP')
                plt.imshow(aux)
                plt.show()

            print('auc :', auc)
            print('jacc :', jacc)
            print('f1 :', f1)
            print('acc :', acc)
            print()

        pd_data['output_file_' + str(suffix)] = row_estimation
        return pd_data

        # pd_data.to_csv(self.save_dir + 'eval_files_'+str(iteration)+'.csv')

        # if 'eval_files_'+str(iteration)+'.csv' not in self.eval_files:
        #     self.eval_files.append('eval_files_'+str(iteration)+'.csv')

    def train_model(self,input_dic,model_trainer_fc,dataset_file='dataset'):

        # data for model
        print('Training model ---- ')
        print('dataset : ',self.save_dir + dataset_file + '.npz')
        npz_file = np.load(self.save_dir + dataset_file + '.npz')
        X_train = npz_file['X_train']
        Y_train = npz_file['Y_train']
        X_val = npz_file['X_val']
        Y_val = npz_file['Y_val']
        print('Basedir, Model name :: ',input_dic['basedir'],input_dic['model_name'])
        save_data = model_trainer_fc([X_train, Y_train, X_val, Y_val], input_dic)

        return save_data

    ## WEIGHTS !!!
    def update_model(self,model_evaluation_fc,model_trainer_fc, epochs=2,
                     save = True, plot = False,load_network = True,
                     tag_X_dir='input_dir', tag_Y_dir='mask_dir',
                     tag_X_file='input_file',tag_Y_file = 'mask_file', iteration = -1):

        if iteration >=0:
            self.iteration = iteration

        # get dataset with from pandas
        self.get_new_dataset(tag_Y_file = 'label_file_'+str(self.iteration),
                             save_file='dataset_it'+str(self.iteration),plots=plot)

        ## Train model
        input_dic = self.model_input_dic.copy()
        input_dic['basedir'] = self.basedir_model
        input_dic['model_name'] = self.model_prefix + 'iteration' + str(self.iteration) + '_'

        if load_network & (len(self.trained_models_dir)>self.iteration):
            input_dic['load_tf_file'] = self.basedir_model +\
                                        self.trained_models_dir[self.iteration-1] + \
                                        '/weights_best.h5'

        input_dic['train_epochs'] = epochs
        save_data = self.train_model(input_dic, model_trainer_fc, dataset_file='dataset_it'+str(self.iteration))

        if len(self.trained_models_dir)> self.iteration:
            self.trained_models_dir[self.iteration] = save_data['model_folder']
        else:
            self.trained_models_dir.append(save_data['model_folder'])
        self.history.append(save_data['history'])

        ## Evaluation
        input_dic['load_tf_file'] = self.basedir_model +\
                                    self.trained_models_dir[self.iteration-1] + \
                                    '/weights_best.h5'

        model_estimator = model_evaluation_fc(input_dic)

        pd_data_eval = self.evaluate(model_estimator, self.eval_data_path,suffix = str(self.iteration),plot=plot,
                                     tag_X_dir = tag_X_dir, tag_Y_dir = tag_Y_dir,
                                     tag_X_file = tag_X_file, tag_Y_file = tag_Y_file)  # outputs a function

        eval_file_name = 'eval_files_'+str(self.iteration)+'.csv'
        pd_data_eval.to_csv(self.save_dir + eval_file_name)
        if len(self.eval_files)> self.iteration:
            self.eval_files[self.iteration] = eval_file_name
        else:
            self.eval_files.append(eval_file_name)

        self.iteration += 1

        #Save
        if save:
            print('SAVING......')
            save_is_dic = self.get_params()
            print(save_is_dic.keys())

            data_save_file = self.save_dir + 'IS_iterations.p'
            print(data_save_file)
            pickle.dump(save_is_dic, open(data_save_file, "wb"))
