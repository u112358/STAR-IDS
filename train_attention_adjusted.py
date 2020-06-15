import argparse
import os.path

from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from tqdm import tqdm

from utils.datareader import *
from utils.dirgen import *
from utils.layers import *


class IDS:
    def __init__(self, shuffle):
        self.shuffle = shuffle
        self.input_dim = 122
        self.nof_class = 9
        self.batch_size = 4000
        self.max_step = 20000
        self.learning_rate = 1e-5
        self.beta1 = 0.5
        self.input_record = tf.placeholder(tf.float32, (None, self.input_dim), name='input_record')
        self.input_label = tf.placeholder(tf.float32, (None, self.nof_class), name='input_label')

        self.intrinsic_f = tf.concat([self.input_record[:, 0:6], self.input_record[:, 39:]], axis=1)
        self.content_f = self.input_record[:, 6:19]
        self.time_f = self.input_record[:, 19:28]
        self.host_f = self.input_record[:, 28:39]

        self.rec_intrinsic = autoencoder(self.intrinsic_f, hidden_layers=[32, 16, 32, 89], name='intrinsic')
        self.rec_content = autoencoder(self.content_f, hidden_layers=[8, 4, 8, 13], name='content')
        self.rec_time = autoencoder(self.time_f, hidden_layers=[8, 4, 8, 9], name='time')
        self.rec_host = autoencoder(self.host_f, hidden_layers=[8, 4, 8, 11], name='host')

        # self.coff1 = regressor(self.input_record, output_dim=89, name='r1')
        # self.coff2 = regressor(self.input_record, output_dim=13, name='r2')
        # self.coff3 = regressor(self.input_record, output_dim=9, name='r3')
        # self.coff4 = regressor(self.input_record, output_dim=11, name='r4')

        self.coff1 = tf.sigmoid(tf.matmul(tf.transpose(self.rec_intrinsic),self.rec_intrinsic))
        self.coff2 = tf.sigmoid(tf.matmul(tf.transpose(self.rec_content), self.rec_content))
        self.coff3 = tf.sigmoid(tf.matmul(tf.transpose(self.rec_time), self.rec_time))
        self.coff4 = tf.sigmoid(tf.matmul(tf.transpose(self.rec_host), self.rec_host))

        self.rec_intrinsic = tf.matmul(self.rec_intrinsic, self.coff1)
        self.rec_content = tf.matmul(self.rec_content, self.coff2)
        self.rec_time = tf.matmul(self.rec_time, self.coff3)
        self.rec_host = tf.matmul(self.rec_host, self.coff4)

        self.reconstructed = tf.concat(
            [self.rec_intrinsic[:, 0:6], self.rec_content, self.rec_time, self.rec_host, self.rec_intrinsic[:, 6:]],
            axis=1)

        self.reconstructed_errors = tf.reduce_sum(tf.squared_difference(self.input_record, self.reconstructed), 1)
        self.loss = tf.reduce_mean(self.reconstructed_errors)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.loss)
        # self.loss_test = tf.reduce_mean(tf.squared_difference(self.input_abnormal, self.reconstructed_abnormal))
        # with tf.name_scope('normal'):
        #     tf.summary.scalar('mean_normal', self.loss, collections=['normal'])
        #     tf.summary.scalar('min_normal', tf.reduce_min(self.reconstructed_errors), collections=['normal'])
        #     tf.summary.scalar('max_normal', tf.reduce_max(self.reconstructed_errors), collections=['normal'])
        #     tf.summary.scalar('var_normal', tf.math.reduce_std(self.reconstructed_errors), collections=['normal'])
        #
        # with tf.name_scope('abnormal'):
        #     tf.summary.scalar('mean_abnormal', self.loss, collections=['abnormal'])
        #     tf.summary.scalar('min_abnormal', tf.reduce_min(self.reconstructed_errors), collections=['abnormal'])
        #     tf.summary.scalar('max_abnormal', tf.reduce_max(self.reconstructed_errors), collections=['abnormal'])
        #     tf.summary.scalar('var_abnormal', tf.math.reduce_std(self.reconstructed_errors), collections=['abnormal'])

        # self.summary_normal = tf.summary.merge_all('normal')
        # self.summary_abnormal = tf.summary.merge_all('abnormal')

        with tf.name_scope('reconstruction_error_statistics'):
            tf.summary.scalar('mean_err', tf.reduce_mean(self.reconstructed_errors))
            tf.summary.scalar('min_err', tf.reduce_min(self.reconstructed_errors))
            tf.summary.scalar('max_err', tf.reduce_max(self.reconstructed_errors))
            tf.summary.scalar('var_err', tf.math.reduce_std(self.reconstructed_errors))
            tf.summary.histogram('hist', self.reconstructed_errors)

        self.summary_op = tf.summary.merge_all()

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_dir = dir_gen('./model_final', __file__)
            saver = tf.train.Saver(max_to_keep=200)
            summary_writer = tf.summary.FileWriter(dir_gen('./log', __file__))
            summary_writer_normal_train = tf.summary.FileWriter(dir_gen('./log/train/normal', __file__))
            summary_writer_abnormal_train = tf.summary.FileWriter(dir_gen('./log/train/abnormal', __file__))
            summary_writer_normal_test = tf.summary.FileWriter(dir_gen('./log/test/normal', __file__))
            summary_writer_abnormal_test = tf.summary.FileWriter(dir_gen('./log/test/abnormal', __file__))
            dr = DataReader('./Data/NSL_TRAIN.mat', self.shuffle)
            for step in tqdm(range(self.max_step)):
                x = dr.get_next_batch(self.batch_size)
                _, loss, sum_normal = sess.run([self.train_op, self.loss, self.summary_op],
                                               feed_dict={self.input_record: x})
                summary_writer.add_summary(sum_normal, step)
                # print("Step: %d\t Loss:%lf" % (step, loss))

                if step % 1000 == 0:
                    x_normal = dr.get_normal_data()
                    sum_normal = sess.run(self.summary_op, feed_dict={self.input_record: x_normal})
                    summary_writer_normal_train.add_summary(sum_normal, step)

                    x_abnormal = dr.get_abnormal_data()
                    sum_abnormal = sess.run(self.summary_op, feed_dict={self.input_record: x_abnormal})
                    summary_writer_abnormal_train.add_summary(sum_abnormal, step)

                    self.check_on_test(sess, summary_writer_normal_test, summary_writer_abnormal_test, step)
                    self.plot_roc_test(sess, step, model_dir)
                if step % 1000 == 0:
                    saver.save(sess, os.path.join(model_dir, 'IDS'), step)
                    # self.save_rec_error(sess, step, model_dir)

    def check_on_test(self, sess, summary_writer_normal_test, summary_writer_abnormal_test, step):
        dr = DataReader('./Data/NSL_TEST.mat')
        x_normal = dr.get_normal_data()
        sum_normal = sess.run(self.summary_op, feed_dict={self.input_record: x_normal})
        summary_writer_normal_test.add_summary(sum_normal, step)
        x_abnormal = dr.get_abnormal_data()
        sum_abnormal = sess.run(self.summary_op, feed_dict={self.input_record: x_abnormal})
        summary_writer_abnormal_test.add_summary(sum_abnormal, step)

    def plot_roc_test(self, sess, step, model_dir):
        dr = DataReader('./Data/NSL_TEST.mat')
        x = dr.get_normal_data()
        score_normal = sess.run(self.reconstructed_errors, feed_dict={self.input_record: x})
        x = dr.get_abnormal_data()
        score_abnormal = sess.run(self.reconstructed_errors, feed_dict={self.input_record: x})
        score = np.concatenate([score_normal, score_abnormal])
        target = np.concatenate([np.zeros_like(score_normal), np.ones_like(score_abnormal)])
        fpr, tpr, _ = roc_curve(y_true=target, y_score=score)
        au = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, marker='.', markersize=2, label='EUIDS')
        # if not os.path.isdir(os.path.join(model_dir, '/plots')):
        #     os.makedirs(os.path.join(model_dir, '/plots'))
        max_acc = find_max_acc(score_normal,score_abnormal)
        fig_path = '%r_roc_%d_auc_%lf_max_acc_%lf.png' % (self.shuffle, step, au,max_acc)
        pyplot.savefig(os.path.join(model_dir, fig_path))
        pyplot.close()

        prec, recall, _ = precision_recall_curve(y_true=target, probas_pred=score)
        # average_precision = average_precision_score(score, target)
        pyplot.plot(recall, prec, marker='.', markersize=2, label='EUIDS')
        # if not os.path.isdir(os.path.join(model_dir, '/plots')):
        #     os.makedirs(os.path.join(model_dir, '/plots'))
        fig_path = '%r_prc_%d.png' % (self.shuffle, step)
        pyplot.savefig(os.path.join(model_dir, fig_path))
        pyplot.close()

    def save_rec_error(self, sess, step, model_dir):
        dr = DataReader('./Data/NSL_TRAIN.mat')
        x = dr.get_normal_data()
        rec_normal = sess.run(self.reconstructed, feed_dict={self.input_record: x})
        x_ = dr.get_abnormal_data()
        rec_abnormal = sess.run(self.reconstructed, feed_dict={self.input_record: x_})
        # if not os.path.isdir(os.path.join(model_dir, '/mats')):
        #     os.makedirs(os.path.join(model_dir, '/mats'))
        filepath = 'train_errors_%d.mat' % step
        filepath = os.path.join(model_dir, filepath)
        sio.savemat(filepath, {'x': x, 'x_': x_, 'rec_normal': rec_normal, 'rec_abnormal': rec_abnormal})

        dr = DataReader('./Data/NSL_TEST.mat')
        x = dr.get_normal_data()
        rec_normal = sess.run(self.reconstructed, feed_dict={self.input_record: x})
        x_ = dr.get_abnormal_data()
        rec_abnormal = sess.run(self.reconstructed, feed_dict={self.input_record: x_})
        # if not os.path.isdir(os.path.join(model_dir, '/mats')):
        #     os.makedirs(os.path.join(model_dir, '/mats'))
        filepath = 'test_errors_%d.mat' % step
        filepath = os.path.join(model_dir, filepath)
        sio.savemat(filepath,
                    {'x': x, 'x_': x_, 'rec_normal': rec_normal, 'rec_abnormal': rec_abnormal})


def find_max_acc(score_normal, score_abnormal):
    size_normal = len(score_normal)
    size_abnormal = len(score_abnormal)
    mid = np.median(score_normal)
    i = np.linspace(0, 2 * mid, 1000)
    acc = []
    for j, th in enumerate(i):
        acc.append((len(np.where(score_normal <= th)[0]) + len(np.where(score_abnormal > th)[0])) / (
                    size_normal + size_abnormal))
    max_acc= np.max(acc)
    return max_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', default=False, type=bool,
                        help='shuffle data set')
    args = parser.parse_args()
    model = IDS(args.shuffle)
    print(args.shuffle)
    model.train()
