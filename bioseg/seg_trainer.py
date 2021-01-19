import numpy as np
from preprocessing import balance_augment_data

def train_model(model_trainer_fc, dataset_path='dataset',
                augment=False, balance=False):
    # data for model
    print('Training model ---- ')
    print('dataset : ', dataset_path)
    npz_file = np.load(dataset_path)
    X_train = npz_file['X_train']
    Y_train = npz_file['Y_train']
    X_val = npz_file['X_val']
    Y_val = npz_file['Y_val']

    if augment:
        X_train, Y_train = balance_augment_data(X_train, Y_train, augment=True, balance=balance)

    print('### SUMMARY DATA ###')
    print('training')
    print(X_train.shape, Y_train.shape)
    print('fraction train patches with annotations : ', np.mean(np.sum(Y_train, axis=(1, 2, 3)) > 0))
    print('Validation')
    print(X_val.shape, Y_val.shape)
    print('fraction val patches with annotations : ', np.mean(np.sum(Y_val, axis=(1, 2, 3)) > 0))

    save_data = model_trainer_fc([X_train, Y_train, X_val, Y_val])

    return save_data

def eval_dataset(model_evaluator, X):
    #X shape [samples,w,h,c]
    Y_eval = []
    for ix in np.arange(X.shape[0]):
        Y_ix = model_evaluator(X[ix,...])
        # print(Y_ix.shape)
        # if ix == 0:
            # Y_eval = np.zeros([X.shape[0], X.shape[1], X.shape[2], Y_ix.shape[-1]])
        Y_eval.append(Y_ix)

    return Y_eval
