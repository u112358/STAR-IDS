import numpy as np
import scipy.io as sio


class DataReader:
    def __init__(self, path, mode, shuffle=True):
        self.mode = mode
        self.shuffle = shuffle
        self.mat = sio.loadmat(path)
        self.data_normal = self.mat['NORMAL']
        self.data_abnormal = np.concatenate(
            [self.mat['DOS'], self.mat['PROBE'], self.mat['R2L'], self.mat['U2R']])
        self.current_index = 0
        self.size_of_normal_data = len(self.data_normal)
        self.size_of_abnormal_data = len(self.data_abnormal)
        self.idx_normal = np.arange(self.size_of_normal_data)
        self.idx_abnormal = np.arange(self.size_of_abnormal_data)
        if shuffle:
            print('shuffling the dataset ...')
            np.random.shuffle(self.idx_abnormal)
            np.random.shuffle(self.idx_normal)
            print('done!')
        self.sub_train, self.sub_val, self.sub_test, self.idx_sub_train, self.current_sub_index = self.split()

    # construct a sub dataset for KDDTrain+ only setting
    def split(self):
        print('constructing subsets from KDDTrain+ ...')
        sub_train = self.data_normal[self.idx_normal[0:53873], :]
        sub_val = np.concatenate([self.data_normal[self.idx_normal[53873:53873 + 6735], :],
                                  self.data_abnormal[self.idx_abnormal[0:6735], :]],
                                 axis=0)
        sub_test = np.concatenate(
            [self.data_normal[self.idx_normal[53873 + 6735:53873 + 6735 + 6735], :],
             self.data_abnormal[self.idx_abnormal[6735:6735 + 6735], :]], axis=0)

        idx_sub_train = np.arange(len(sub_train))
        np.random.shuffle(idx_sub_train)
        current_sub_index = 0
        print('done!')
        return sub_train, sub_val, sub_test, idx_sub_train, current_sub_index

    # training use only the normal data
    def get_next_batch(self, batch_size):
        if self.mode == 'test+':
            if self.current_index + batch_size > self.size_of_normal_data:
                self.current_index = 0
                np.random.shuffle(self.idx_normal)
            if self.shuffle:
                _x = self.data_normal[self.idx_normal[self.current_index:self.current_index + batch_size], :]
            else:
                _x = self.data_normal[self.current_index:self.current_index + batch_size, :]
            self.current_index += batch_size
            return _x
        elif self.mode == 'train+':
            if self.current_sub_index + batch_size > len(self.sub_train):
                self.current_sub_index = 0
                np.random.shuffle(self.idx_sub_train)
            if self.shuffle:
                _x = self.sub_train[
                     self.idx_sub_train[self.current_sub_index:self.current_sub_index + batch_size], :]
            else:
                _x = self.sub_train[self.current_sub_index:self.current_sub_index + batch_size, :]
            self.current_sub_index += batch_size
            return _x
        else:
            assert 'unknown mode!'

    # get all abnormal data for validation
    def get_abnormal_data(self):
        return self.data_abnormal

    # get all normal data for validation
    def get_normal_data(self):
        return self.data_normal

    # get the whole sub training set
    def get_train(self):
        return self.sub_train

    # get data for validation
    def get_validation(self, label='all'):
        if label == 'normal':
            return self.sub_val[0:6735, :]
        elif label == 'abnormal':
            return self.sub_val[6735:, :]
        elif label == 'all':
            return self.sub_val
        else:
            return self.sub_val

    # get data for test
    def get_test(self, label='all'):
        if label == 'normal':
            return self.sub_test[0:6735, :]
        elif label == 'abnormal':
            return self.sub_test[6735:, :]
        elif label == 'all':
            return self.sub_test
        else:
            return self.sub_test
