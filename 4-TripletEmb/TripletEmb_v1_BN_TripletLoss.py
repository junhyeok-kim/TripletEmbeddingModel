# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
import math
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class TripletEmbedding(object):
    
    def __init__(self,
                 d,
                 k,
                 l,
                 C,
                 init_width,
                 margin,
                 min_lr,
                 max_lr,
                 keep_ratio,
                 phase,
                 method,
                 batch_size,
                 epoch,
                 stop_criterion,
                 lambda_value,
                 session,
                 result_save_path,
                 log_save_path,
                 log_save_filename):
        """
         To differentiate Class instances and Method's instances,
         we use 'self._' and 'self.'
        """
        self._d = d
        self._k = k
        self._l = l
        self._C = C
        self._init_width = init_width
        self._margin = margin
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._dropout = keep_ratio
        self._phase = phase
        self._method = method
        self._batch_size = batch_size
        self._epoch = epoch
        self._stop_criterion = stop_criterion
        self._lambda_value = lambda_value
        self._session = session
        self._result_save_path = result_save_path
        self._log_save_path = log_save_path
        self._log_save_filename = log_save_filename
        self.build_graph()
        
    def my_logger(self, logger_name, log_save_path, log_save_filename):
        """ print and save some information """
        logger = logging.getLogger(logger_name)
        file_handler = logging.FileHandler(log_save_path + '/' + log_save_filename + '.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s ')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def BilinearTensorProduct(self, e1, R, e2):
        """get quadratic form of vectors: t(e1)T^[1:k]e2"""
        e1 = tf.expand_dims(e1, 1)
        e2 = tf.expand_dims(e2, 1)
        op_val = tf.einsum('nij,kjl->nkl', e1, R)
        quadratic = tf.einsum('nkl,nml->nk', op_val, e2)
        return quadratic
    
    def NTNLayer(self, e1, e2, e1e2, T, W, b):
        """ NTN-based function: Bilinear form (Socher et al. 2013) """
        quadratic = self.BilinearTensorProduct(e1, T, e2)
        e1e2 = tf.expand_dims(e1e2, 1)
        op_val2 = tf.einsum('ij,nki->nj', W, e1e2)
        op_val3 = tf.add(tf.add(quadratic, op_val2), b)
        tanh_input = tf.layers.batch_normalization(
                            op_val3,
                            center=True,
                            scale=True,
                            training=self._phase,
                            trainable=True,
                            reuse=None)
        out = tf.tanh(tanh_input)
        return tanh_input, out

    def EELM(self, R1, R2, R1R2, T3, W3, b3, mu):
        """ get score of each SVO """
        EEV2, U = self.NTNLayer(R1, R2, R1R2, T3, W3, b3)
        score = tf.einsum('ij,kj->i', U, mu)
        return EEV2, U, score
    
    def TripletLoss(self, anchor, positive, negative,
                    T1, T2, T3, W1, W2, W3, mu):
        """ Triplet Loss inspired by Schroff et al.(2015) 
        Here,
            anchor = real SVO
            positive = corrupted S'VO
            negative = corrupted SVO'
        """
        ap_l2 = tf.square(tf.subtract(anchor, positive))
        an_l2 = tf.square(tf.subtract(anchor, negative))
        each_loss = tf.maximum(tf.constant(0., dtype=tf.float32),
                           tf.add(tf.subtract(ap_l2, an_l2),
                                  tf.constant(self._margin, dtype=tf.float32)))
        margin_loss = tf.reduce_sum(each_loss)
        regularizer = tf.multiply(tf.constant(self._lambda_value, dtype=tf.float32),
                                  tf.add_n([tf.nn.l2_loss(T1),
                                            tf.nn.l2_loss(T2),
                                            tf.nn.l2_loss(T3),
                                            tf.nn.l2_loss(W1),
                                            tf.nn.l2_loss(W2),
                                            tf.nn.l2_loss(W3),
                                            tf.nn.l2_loss(mu)]))
        loss = tf.add(margin_loss, regularizer)
                         
        return margin_loss, loss
    
    def optimize(self, lr, loss, method = 'GD'):
        """ optimizer """
        if method == 'GD':
            print("Preparing optimization by Gradient Descent with CLR")
            return tf.train.GradientDescentOptimizer(lr).minimize(loss)
        elif method == 'AdaGrad':
            print("Preparing optimization by AdaGrad")
            return tf.train.AdagradOptimizer(lr).minimize(loss)
        elif method == 'RMSProp':
            print("Preparing optimization by RMSProp")
            return tf.train.RMSPropOptimizer(lr).minimize(loss)
        elif method == 'Adam':
            print("Preparing optimization by Adam")
            return tf.train.AdamOptimizer(lr).minimize(loss)
        
    def get_EventEmbMatrix(self, vecS, vecV, vecO, size):
        """ Make EEV:
            For the GPU Memory problem("InternalError: Dst tensor is not initialized.")
            calculate it by slicing
        """
        tmp = np.arange(vecS.shape[0], step=size)
        EEV = np.zeros((vecS.shape[0], self._l))
        EEV2 = np.zeros((vecS.shape[0], self._l))
        
        for i, start_val in enumerate(tmp):
            if i == (len(tmp)-1):
                end_val = vecS.shape[0]
                feed_dict = {self.s_input: vecS[start_val:end_val,:],
                             self.v_input: vecV[start_val:end_val,:],
                             self.o_input: vecO[start_val:end_val,:]}
                EEV_vec = self._session.run(self.EEV, feed_dict = feed_dict)
                EEV[start_val:end_val,:] = EEV_vec
                EEV_vec = self._session.run(self.EEV2, feed_dict = feed_dict)
                EEV2[start_val:end_val,:] = EEV_vec
            else:
                end_val = start_val + size
                feed_dict = {self.s_input: vecS[start_val:end_val,:],
                             self.v_input: vecV[start_val:end_val,:],
                             self.o_input: vecO[start_val:end_val,:]}
                EEV_vec = self._session.run(self.EEV, feed_dict = feed_dict)
                EEV_vec = self._session.run(self.EEV, feed_dict = feed_dict)
                EEV[start_val:end_val,:] = EEV_vec
                EEV_vec = self._session.run(self.EEV2, feed_dict = feed_dict)
                EEV2[start_val:end_val,:] = EEV_vec

        return EEV, EEV2

    def generate_batch_w_corrupted(self, vecS, vecV, vecO,
                                   batch_size, C, idx_used):
        """ (C*batch_size, d): input data
        - randomly selected subject, make sure without replacement (batch)
        (although, replacement is also possible)
        - after loading mini-batch numpy array, e.g. batch_size = 32, (32, d)
        1) generate corrupted data for 'each' input data e.g. C=5, (32*5, d)
        2) extending input data with same dim of corrupted data e.g. (32*5, d)    
    
        """
        batch_s = vecS[idx_used,:]
        batch_v = vecV[idx_used,:]
        batch_o = vecO[idx_used,:]
        
        idx_c = np.random.choice(vecS.shape[0], size=batch_size*C, replace=False)
        batch_sc = vecS[idx_c, ]
        idx_c = np.random.choice(vecO.shape[0], size=batch_size*C, replace=False)
        batch_oc = vecO[idx_c, ]
        
        stack_s = np.copy(batch_s)
        stack_v = np.copy(batch_v)
        stack_o = np.copy(batch_o)
        for i in range(C-1):
            batch_s = np.vstack((batch_s, stack_s))
            batch_v = np.vstack((batch_v, stack_v))
            batch_o = np.vstack((batch_o, stack_o))
            
        return batch_s, batch_v, batch_o, batch_sc, batch_oc
          
    def loss_plot(self, total_loss, total_margin_loss):
        total_loss = np.array(total_loss)
        total_margin_loss = np.array(total_margin_loss)

        plt.figure(figsize=(20,10))
        plt.plot(np.arange(len(total_loss)), total_loss)
        plt.plot(np.arange(len(total_margin_loss)), total_margin_loss)
        plt.title("Average Epoch Loss changes in every epochs", fontsize=20)
        plt.xlabel('Number of Epochs', fontsize=20)
        plt.savefig(self._result_save_path + '/epoch_loss.png')
    
    def build_graph(self):
        """Build the graph"""
        # read hyperparameters from class instances
        d = self._d
        k = self._k
        l = self._l
        
        # dropout
        s_input   = tf.placeholder(tf.float32, shape=(None, d), name='s_input')
        v_input   = tf.placeholder(tf.float32, shape=(None, d), name='v_input')
        o_input   = tf.placeholder(tf.float32, shape=(None, d), name='o_input')
        sv_input  = tf.concat([s_input, v_input], 1)
        vo_input  = tf.concat([v_input, o_input], 1)     
        # corrupted placeholder
        sc_input    = tf.placeholder(tf.float32, shape=(None, d))
        sc_v_input  = tf.concat([sc_input, v_input], 1)
        oc_input    = tf.placeholder(tf.float32, shape=(None, d))
        v_oc_input  = tf.concat([v_input, oc_input], 1)
        # variables        
        init_T = tf.random_uniform((k, d, d), -self._init_width, self._init_width,
                                   dtype=tf.float32)
        init_W = tf.random_uniform((2*d, k), -self._init_width, self._init_width,
                                   dtype=tf.float32)
        init_b = tf.random_uniform((1, k), -self._init_width, self._init_width,
                                   dtype=tf.float32)
                    
        with tf.name_scope('layer_first'):
            T1 = tf.Variable(init_T, name="Tensor1")
            W1 = tf.Variable(init_W, name="Weight1")
            b1 = tf.Variable(init_b, name="bias1")
            
        with tf.name_scope('layer_second'):
            T2 = tf.Variable(init_T, name="Tensor2")
            W2 = tf.Variable(init_W, name="Weight2")
            b2 = tf.Variable(init_b, name="bias2")
                
        with tf.name_scope('layer_third'):
            T3 = tf.Variable(
                    tf.random_uniform((l, k, k), -self._init_width, self._init_width,
                                      dtype=tf.float32), name="Tensor3")
            W3 = tf.Variable(
                    tf.random_uniform((2*k, l), -self._init_width, self._init_width,
                                      dtype=tf.float32), name="Weight3")
            b3 = tf.Variable(
                    tf.random_uniform((1, l), -self._init_width, self._init_width,
                                               dtype=tf.float32), name="bias3")
            mu = tf.Variable(
                    tf.random_uniform((1, l), -self._init_width, self._init_width,
                                      dtype=tf.float32), name="score_params")

        # Real event score and EEV : f(E_{i}) 
        _, R1 = self.NTNLayer(s_input, v_input, sv_input, T1, W1, b1)
        _, R2 = self.NTNLayer(v_input, o_input, vo_input, T2, W2, b2)
        R1R2 = tf.concat([R1, R2], 1)
        EEV2, EEV, event_score = self.EELM(R1, R2, R1R2, T3, W3, b3, mu)
        EEV = tf.identity(EEV, name='event_embedding_vector')
        EEV2 = tf.identity(EEV2, name='event_embedding_vector2')
        self.event_score = event_score
        
        # S Corrupted event Score : f(E^r_{i})
        _, R1_sc = self.NTNLayer(sc_input, v_input, sc_v_input, T1, W1, b1)
        R1R2_sc = tf.concat([R1_sc, R2], 1)
        _, _, s_corrupted_score = self.EELM(R1_sc, R2, R1R2_sc, T3, W3, b3, mu)
        self.s_corrupted_score = s_corrupted_score
        
        # O Corrupted event Score : f(E^r_{i})
        _, R2_oc = self.NTNLayer(v_input, o_input, v_oc_input, T2, W2, b2)
        R1R2_oc = tf.concat([R1, R2_oc], 1)
        _, _, o_corrupted_score = self.EELM(R1, R2_oc, R1R2_oc, T3, W3, b3, mu)
        self.o_corrupted_score = o_corrupted_score

        # loss function
        margin_loss, loss = self.TripletLoss(
                        event_score, s_corrupted_score, o_corrupted_score,
                        T1, T2, T3, W1, W2, W3, mu)
        # optimizer
        lr = tf.placeholder(tf.float32)
        self.lr = lr
        optimizer = self.optimize(lr, loss, method=self._method)
        
        # method instances
        self.s_input = s_input
        self.v_input = v_input
        self.o_input = o_input
        self.sc_input = sc_input
        self.oc_input = oc_input
        self.update = optimizer
        self.margin_loss = margin_loss
        self.loss = loss
        # final output: Event Embedding Vectors
        self.EEV = EEV   
        self.EEV2 = EEV2   
        self.saver = tf.train.Saver()
        print('----------------- All Trainable Parameters -----------------')
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)   # i.name if you want just a name
    
    def train(self, vecS, vecV, vecO):
        """ Train the EELM
        training method:
            Socher(2013): Mini-batched L-BFGS > AdaGrad
            Ding(2015)  : Standard backpropagation
        """
        log_save_path = self._log_save_path
        log_save_filename = self._log_save_filename
        logger = self.my_logger('JUNHYEOK KIM', log_save_path, log_save_filename)

        logger.info('Global Variables are Initialized')
        self._session.run(tf.global_variables_initializer())
        start_time = time.time()
      
        # for cyclical learning rate
        data_size = vecS.shape[0]
        max_iter = math.ceil(data_size/self._batch_size)
        min_lr = self._min_lr
        max_lr = self._max_lr
        step_size = max_iter / 2
        iter_count = 0
        
        total_loss = []
        total_margin_loss = []

        for N in range(self._epoch):
            batch_size = self._batch_size
            idx_set = np.arange(vecS.shape[0])
            iter_num = math.ceil(vecS.shape[0] / batch_size)
            
            # tracking
            epoch_loss = []
            epoch_margin_loss = []

            for i in range(iter_num):
                if idx_set.size == 1:
                    break
                elif idx_set.size < batch_size:
                    batch_size = idx_set.size
                else:
                    pass
                
                # cyclical learning rate
                cycle = math.floor(1 + iter_count/(2*step_size))
                x = abs(iter_count / step_size - 2*cycle + 1)
                lr = min_lr + (max_lr - min_lr)*max(0, (1 - x))
                iter_count += 1
        
                idx_used = np.random.choice(idx_set, size=batch_size,
                                            replace=False)
                batch_s, batch_v, batch_o, batch_sc, batch_oc = self.generate_batch_w_corrupted(
                                                         vecS, vecV, vecO,
                                                         batch_size, self._C,
                                                         idx_used)
        
                feed_batch = {self.s_input: batch_s,
                              self.v_input: batch_v,
                              self.o_input: batch_o,
                              self.sc_input: batch_sc,
                              self.oc_input: batch_oc,
                              self.lr: lr}
        
                # update parameters
                _, loss_val, marginloss_val = self._session.run(
                        [self.update, self.loss, self.margin_loss],
                        feed_dict = feed_batch)

                if (i+1) % 200 == 0:
                    logger.info("------------------------------------------------")
                    logger.info("learning rate: " + str(lr))
                    logger.info("Epoch Number is " + str(N+1) +
                                " / Item number is " + str(i+1))
                    logger.info(str(loss_val / (batch_size*self._C)) + " / " +
                                str(marginloss_val / (batch_size*self._C)))
                    
                # save the loss
                epoch_loss.append(loss_val / (batch_size*self._C))
                epoch_margin_loss.append(marginloss_val / (batch_size*self._C))
                
                idx_set = np.setdiff1d(idx_set, idx_used)

        # ---------------------------------------------------------------------
            logger.info("=========== " + str(N+1) + " Epoch is done ===========")
            logger.info(str(N+1) + " Epoch loss: " + str(np.sum(epoch_loss) / iter_num))
            logger.info(str(N+1) + " Epoch Margin loss: " + str(np.sum(epoch_margin_loss) / iter_num))

            # -----------------------------------------------------------------
            logger.info(str(N+1) + " epoch time is: "
                        + str((time.time() - start_time)/60) + " minutes")
            # save the epoch loss
            total_loss.append(np.sum(epoch_loss) / iter_num)     
            total_margin_loss.append(np.sum(epoch_margin_loss) / iter_num)
        # ---------------------------------------------------------------------
        # early stopping
            if (np.sum(epoch_loss) / iter_num) <= self._stop_criterion:
                logger.info("Saving the EEV...")
                EventEmbMatrix, EventEmbMatrix2 = self.get_EventEmbMatrix(
                        vecS, vecV, vecO, size=256)
                np.save(self._result_save_path
                        + '/EEV_v1_e{}_d{}_k{}_l{}_C{}.npy'.format(N+1, self._d, self._k, self._l, self._C),
                        EventEmbMatrix)
                np.save(self._result_save_path
                        + '/EEV2_v1_e{}_d{}_k{}_l{}_C{}.npy'.format(N+1, self._d, self._k, self._l, self._C),
                        EventEmbMatrix2)
                logger.info("Saving the model...")
                self.saver.save(self._session,
                                self._result_save_path
                                + '/EventEmb_v1_e{}_d{}_k{}_l{}_C{}.ckpt'.format(N+1, self._d, self._k, self._l, self._C))
                
                logger.info("Final Execution time is: " 
                            + str((time.time() - start_time)/60) + " minutes")
                # save the loss
                self.loss_plot(total_loss, total_margin_loss)
                np.save(self._result_save_path + '/total_loss.npy', np.array(total_loss))
                np.save(self._result_save_path + '/total_margin_loss.npy', np.array(total_margin_loss))
                logger.info("complete: saved final model")
                break
    # =========================================================================            
        logger.info("Final Execution time is: " 
                    + str((time.time() - start_time)/60) + " minutes")
        logger.info("Saving the EEV...")
        EventEmbMatrix, EventEmbMatrix2 = self.get_EventEmbMatrix(
                vecS, vecV, vecO, size=256)
        np.save(self._result_save_path
                + '/EEV_v1_e{}_d{}_k{}_l{}_C{}.npy'.format(N+1, self._d, self._k, self._l, self._C),
                EventEmbMatrix)
        np.save(self._result_save_path
                + '/EEV2_v1_e{}_d{}_k{}_l{}_C{}.npy'.format(N+1, self._d, self._k, self._l, self._C),
                EventEmbMatrix2)
        logger.info("Saving the model...")
        self.saver.save(self._session,
                        self._result_save_path
                        + '/EventEmb_v1_e{}_d{}_k{}_l{}_C{}.ckpt'.format(N+1, self._d, self._k, self._l, self._C))
        self.loss_plot(total_loss, total_margin_loss)
        np.save(self._result_save_path + '/total_loss.npy', np.array(total_loss))
        np.save(self._result_save_path + '/total_margin_loss.npy', np.array(total_margin_loss))
        logger.info("complete: saved final model")
    # =========================================================================


if __name__ == "__main__":
    print("Ready to Build the graph for NTN")