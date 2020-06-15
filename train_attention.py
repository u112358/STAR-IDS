import argparse
import os.path
import tensorflow as tf

from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from tqdm import tqdm

from utils.datareader import *
from utils.dirgen import *
from utils.layers import *


class STARIDS:
    """
    STAR-IDS is an auto encoder based model for intrusion detection, in which the original input
    network traffic record is split into sub features to explore the intrinsic structure of the network
    traffic record
    """

    def __init__(self, shuffle):
        self.shuffle = shuffle  # flag indicating if shuffle the dataset or not
        self.input_dim = 122  # dimension of input feature
        self.nof_class = 9  # categories
        self.batch_size = 4000  # batch size
        self.max_step = 60000  # max step for training the model
        self.learning_rate = 1e-5  # learning rate
        self.beta1 = 0.5  # parameter for Adam Optimizer
        self.input_record = tf.placeholder(tf.float32, (None, self.input_dim), name='input_record')  # input feature
        # self.input_label = tf.placeholder(tf.float32, (None, self.nof_class),
        #                                   name='input_label')  # input label (one hot)

        # here we split the feature into four sub categories
        # which are intrinsic feature, content feature, time feature and host feature
        self.intrinsic_f = tf.concat([self.input_record[:, 0:6], self.input_record[:, 39:]], axis=1)
        self.content_f = self.input_record[:, 6:19]
        self.time_f = self.input_record[:, 19:28]
        self.host_f = self.input_record[:, 28:39]

        # define the reconstructed features
        # note that each auto encoder has its own name scope therefore the structures are different
        self.rec_intrinsic = autoencoder(self.intrinsic_f, hidden_layers=[32, 16, 32, 89], name='intrinsic')
        self.rec_content = autoencoder(self.content_f, hidden_layers=[8, 4, 8, 13], name='content')
        self.rec_time = autoencoder(self.time_f, hidden_layers=[8, 4, 8, 9], name='time')
        self.rec_host = autoencoder(self.host_f, hidden_layers=[8, 4, 8, 11], name='host')

        # the reconstructed feature are concatenated to match the original feature
        self.reconstructed = tf.concat(
            [self.rec_intrinsic[:, 0:6], self.rec_content, self.rec_time, self.rec_host, self.rec_intrinsic[:, 6:]],
            axis=1)

        # reconstructed error
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
        gpu_config = tf.ConfigProto(allow_soft_placement=False)
        with tf.Session(config=gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            model_dir = dir_gen('./model_att', __file__)
            saver = tf.train.Saver(max_to_keep=200, var_list=tf.all_variables())
            summary_writer = tf.summary.FileWriter(dir_gen('./log', __file__))
            summary_writer_normal_train = tf.summary.FileWriter(dir_gen('./log/train/normal', __file__))
            summary_writer_abnormal_train = tf.summary.FileWriter(dir_gen('./log/train/abnormal', __file__))
            summary_writer_normal_test = tf.summary.FileWriter(dir_gen('./log/test/normal', __file__))
            summary_writer_abnormal_test = tf.summary.FileWriter(dir_gen('./log/test/abnormal', __file__))
            dr = DataReader('./Data/NSL_TRAIN.mat', mode='train+', shuffle=self.shuffle)
            rsl = []
            for step in tqdm(range(self.max_step)):
                x = dr.get_next_batch(self.batch_size)
                _, loss, sum_normal = sess.run([self.train_op, self.loss, self.summary_op],
                                               feed_dict={self.input_record: x})
                summary_writer.add_summary(sum_normal, step)
                # print("Step: %d\t Loss:%lf" % (step, loss))

                if step % 500 == 0:
                    x_normal = dr.get_validation()
                    sum_normal = sess.run(self.summary_op, feed_dict={self.input_record: x_normal})
                    summary_writer_normal_train.add_summary(sum_normal, step)

                    x_abnormal = dr.get_abnormal_data()
                    sum_abnormal = sess.run(self.summary_op, feed_dict={self.input_record: x_abnormal})
                    summary_writer_abnormal_train.add_summary(sum_abnormal, step)

                    self.check_on_test(sess, dr, summary_writer_normal_test, summary_writer_abnormal_test, step)
                    self.plot_roc_test(sess, dr, step, model_dir, rsl)

                if step % 500 == 0:
                    saver.save(sess, os.path.join(model_dir, 'IDS'), step)
                    # self.save_rec_error(sess, step, model_dir)
            rsl_path = os.path.join(model_dir, 'rsl.mat')
            sio.savemat(rsl_path, {'result': rsl})
            self.plot_rsl(rsl, model_dir)

    def plot_rsl(self, rsl, model_dir):
        step=[]
        max_acc=[]
        prec=[]
        recall=[]
        for i in range(len(rsl)):
            step.append(rsl[i][0])
            max_acc.append(rsl[i][1])
            prec.append(rsl[i][2])
            recall.append(rsl[i][3])
        pyplot.plot(step, max_acc, marker='.', markersize=2)
        pyplot.plot(step, prec, marker='o', linestyle='-')
        pyplot.plot(step, recall, marker='*', linestyle='-')
        pyplot.legend(('accuracy', 'precision', 'recall'), loc='upper right')
        pyplot.savefig(os.path.join(model_dir, 'result.png'))
        pyplot.close()

    def check_on_test(self, sess, dr, summary_writer_normal_test, summary_writer_abnormal_test, step):
        # x_normal = dr.get_normal_data()
        x_normal = dr.get_test('normal')
        sum_normal = sess.run(self.summary_op, feed_dict={self.input_record: x_normal})
        summary_writer_normal_test.add_summary(sum_normal, step)
        # x_abnormal = dr.get_abnormal_data()
        x_abnormal = dr.get_test('abnormal')
        sum_abnormal = sess.run(self.summary_op, feed_dict={self.input_record: x_abnormal})
        summary_writer_abnormal_test.add_summary(sum_abnormal, step)

    def plot_roc_test(self, sess, dr, step, model_dir, rsl):
        x = dr.get_test('normal')
        score_normal = sess.run(self.reconstructed_errors, feed_dict={self.input_record: x})
        x = dr.get_test('abnormal')
        score_abnormal = sess.run(self.reconstructed_errors, feed_dict={self.input_record: x})
        score = np.concatenate([score_normal, score_abnormal])
        target = np.concatenate([np.zeros_like(score_normal), np.ones_like(score_abnormal)])
        fpr, tpr, _ = roc_curve(y_true=target, y_score=score)
        au = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, marker='.', markersize=2, label='EUIDS')
        # if not os.path.isdir(os.path.join(model_dir, '/plots')):
        #     os.makedirs(os.path.join(model_dir, '/plots'))
        max_acc, max_index, num_th, best_threshold = find_max_acc(score_normal, score_abnormal)
        fig_path = '%r_roc_%d_auc_%lf_max_acc_%lf_at%dof%d.png' % (self.shuffle, step, au, max_acc, max_index, num_th)
        pyplot.savefig(os.path.join(model_dir, fig_path))
        pyplot.close()

        prec, recall, threshold = precision_recall_curve(y_true=target, probas_pred=score)
        # average_precision = average_precision_score(score, target)
        pyplot.plot(recall, prec, marker='.', markersize=2, label='EUIDS')
        # if not os.path.isdir(os.path.join(model_dir, '/plots')):
        #     os.makedirs(os.path.join(model_dir, '/plots'))
        index = np.argmin(np.abs(threshold-best_threshold))
        print('prec: %lf \t recall: %lf \n' % (prec[index], recall[index]))
        fig_path = '%r_prc_%d_prec_%lf_rec_%lf.png' % (self.shuffle, step, prec[index], recall[index])
        pyplot.savefig(os.path.join(model_dir, fig_path))
        pyplot.close()

        rsl.append([step, max_acc, prec[index], recall[index], index, num_th])

    def eval(self, model_dir):
        with tf.Session() as sess:
            saver = tf.train.Saver(var_list=tf.all_variables())
            saver.restore(sess, save_path=model_dir)
            dr = DataReader('./Data/NSL_TRAIN.mat', mode='train+')
            x = dr.get_test('normal')
            score_normal = sess.run(self.reconstructed_errors, feed_dict={self.input_record: x})
            x = dr.get_test('abnormal')
            score_abnormal = sess.run(self.reconstructed_errors, feed_dict={self.input_record: x})
            score = np.concatenate([score_normal, score_abnormal])
            target = np.concatenate([np.zeros_like(score_normal), np.ones_like(score_abnormal)])
            max_acc, max_idx, num_th, best_threshold = find_max_acc(score_normal, score_abnormal)
            prec, recall, threshold = precision_recall_curve(y_true=target, probas_pred=score)
            index = np.argmin(np.abs(threshold-best_threshold))
            print('prec: %lf \t recall: %lf \n' % (prec[index], recall[index]))
            print('f1:%lf' %(prec[index]*recall[index]*2/(prec[index]+recall[index])))


def find_max_acc(score_normal, score_abnormal):
    size_normal = len(score_normal)
    size_abnormal = len(score_abnormal)
    mid = np.median(score_normal)
    i = np.arange(0, 2 * mid, 0.001)
    acc = []
    for j, th in enumerate(i):
        acc.append((len(np.where(score_normal <= th)[0]) + len(np.where(score_abnormal > th)[0])) / (
                size_normal + size_abnormal))
    max_acc = np.max(acc)
    max_index = int(np.argmax(acc))
    best_threshold = i[max_index]
    num_th = len(acc)
    print("maximal accuracy:%lf is yielded at %d of total %d thresholds" % (max_acc, max_index, num_th))
    return max_acc, max_index, num_th, best_threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', default=False, type=bool,
                        help='shuffle data set')
    args = parser.parse_args()
    model = STARIDS(args.shuffle)
    model.train()
