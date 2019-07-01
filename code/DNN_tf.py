import tensorflow as tf
import numpy as np
from generate_dateSet import generate_Dataset


class NN:
    def __init__(self, leaning_rate=3e-4, max_iterators=20000, batch_size=200):
        self.learning_rate = leaning_rate
        self.max_iterators = max_iterators
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, 1364])
        self.y = tf.placeholder(tf.float32, [None, 2])
        self.dataset_size = 0
        self.length = 0
        self.input_layer = 0


    def Normalization(self,x):
        m = min(x)
        M = max(x)
        return  [ (float(i)-m) / float(M - m) for i in x  ]
    def get_data(self):
        d = generate_Dataset("data/firefox.txt")
        ben_data, neg_data = d.getData()
        length = d.getMaxGadget()
        self.length = length

        l1 = len(ben_data)
        l2 = len(neg_data)
        x = ben_data + neg_data
        #x = self.Normalization(x)  # Normalization
        ben_data = x[:l1]
        neg_data = x[l1:]

        #
        ben_data = np.array(ben_data).reshape([int(len(ben_data) / (length - 10)),length - 10])
        neg_data = np.array(neg_data).reshape([int(len(neg_data) / (length - 10)), length - 10])
        ben_data = ben_data[:-1]
        neg_data = neg_data[:-1]
        self.dataset_size = len(ben_data) + len(neg_data)

        x = []
        for i in range(len(ben_data)):
            x.append(ben_data[i])
        for i in range(len(neg_data)):
            x.append(neg_data[i])

        y = []
        for i in range(len(ben_data)):
            y.append([1.0,0.0])
        for i in range(len(neg_data)):
            y.append([0.0,1.0])

        index = [i for i in range(self.dataset_size)]  #shuffle
        #np.random.seed(10000)
        np.random.shuffle(index)
        index = np.array(index)
        images = np.array(x)[index]
        labels = np.array(y)[index]
        print("len labels :", len(labels), "index :", labels)

        images = np.array(images)
        labels = np.array(labels).reshape([len(labels),2])
        print("len x:",len(images[:1]))
        print("x:",images[1:2])
        print("x shape:" ,images.shape)
        print("y shape:" ,labels.shape)
        print("length of dataset: %d" % (len(images)))
        print("length of dataset: %d" % (len(labels)))
        self.input_layer = images.shape[1]
        print("input_shape:",self.input_layer)
        #self.x ,self.y = x , y
        return images,labels

    def DNN(self):
        #raise NotImplementedError('Method inference is not implemented.')
        x = self.x

        W1 = self.weight_variables([self.input_layer, 1024])
        b1 = self.biases_variables([1024])
        x = tf.matmul(x, W1) + b1
        x = tf.nn.relu(x)

        W2 = self.weight_variables([512, 128])
        b2 = self.biases_variables([128])
        x = tf.matmul(x, W2) + b2
        x = tf.nn.relu(x)

        W3 = self.weight_variables([128, 32])
        b3 = self.biases_variables([32])
        x = tf.matmul(x, W3) + b3
        x = tf.nn.relu(x)

        W4 = self.weight_variables([32, 2])
        b4 = self.biases_variables([2])
        x = tf.matmul(x, W4) + b4
        #x = tf.nn.softmax(x)
        return x

    def LR(self):
        x = self.x
        W = self.weight_variables([self.input_layer, 2])
        b = self.biases_variables([2])
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        return y

    def LR_train_step(self, logits, labels):
        # LR  Cost function
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-10), reduction_indices=1))
        return optimizer.minimize(cost)

    def DNN_train_step(self,logits, labels):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return optimizer.minimize(loss)


    def train(self):
        x, y = self.get_data()
        self.x = tf.placeholder(tf.float32, [None, self.input_layer])
        self.y = tf.placeholder(tf.float32, [None, 2])


        #prediction = self.DNN()
        prediction = self.LR()

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        #train_op = self.DNN_train_step(prediction, self.y)
        train_op = self.LR_train_step(prediction, self.y)

        fps,false_positive = tf.metrics.false_positives( labels=tf.argmax(self.y, 1),predictions=tf.argmax(prediction, 1))
        fng,false_negative = tf.metrics.false_negatives( labels=tf.argmax(self.y, 1),predictions=tf.argmax(prediction, 1))
        rcl,recall = tf.metrics.recall(labels=tf.argmax(self.y, 1),predictions=tf.argmax(prediction, 1))
        pre,precision = tf.metrics.precision(labels=tf.argmax(self.y, 1),predictions=tf.argmax(prediction, 1))
        aucc,auc = tf.metrics.auc(labels=tf.argmax(self.y, 1),predictions=tf.argmax(prediction, 1))
        accy,accu = tf.metrics.accuracy(labels=tf.argmax(self.y, 1),predictions=tf.argmax(prediction, 1))

        #here

        train_x = x[:int(4*len(x)/5)]
        train_y = y[:int(4 * len(y) / 5)]
        #test_x = x[int(4*len(x)/5):]#int(6*len(x)/7)
        test_x = x[:]
        #test_y = y[int(4 * len(y) / 5):]#int(6*len(y)/7)
        test_y = y[:]

        print("train_x shape : ",train_x.shape)
        print("train_y shape : ", train_y.shape)
        print(type(train_x))
        self.saver = tf.train.Saver(max_to_keep=1)  # save model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            batch_size = self.batch_size
            dataset_size = self.dataset_size
            for step in range(1, self.max_iterators + 1):
                start = (step * batch_size) % dataset_size
                end = min(start + batch_size, dataset_size)
                batch_x, batch_y = train_x[start:end],train_y[start:end]
                sess.run(train_op, feed_dict={self.x: batch_x, self.y: batch_y})
                if step <= 10 or step % 1000 == 0 or step == self.max_iterators:
                    acc = sess.run(accuracy, feed_dict={self.x: test_x, self.y: test_y})
                    print('Step %s, Accuracy %s' % (step, acc))
            fp = sess.run(false_positive,feed_dict={self.x: test_x, self.y: test_y})
            fn = sess.run(false_negative,feed_dict={self.x: test_x, self.y: test_y})
            rc = sess.run(recall, feed_dict={self.x: test_x, self.y: test_y})
            pc = sess.run(precision, feed_dict={self.x: test_x, self.y: test_y})
            #au = sess.run(auc, feed_dict={self.x: test_x, self.y: test_y})
            ac = sess.run(accu, feed_dict={self.x: test_x, self.y: test_y})
            print("false_positives : %d / %d percentage = %f " % (fp,len(test_y),fp/len(test_y)*100))
            print("false_negatives:%d / %d percentage= %f " % (fn,len(test_y),fn/len(test_y)*100))
            print("recall :", rc)
            print("precision :", pc)
            #print("auc :", au)
            print("accuracy :", ac)
            self.save(sess)


    def weight_variables(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def biases_variables(self, shape):
        return tf.Variable(tf.constant(0.01, shape=shape))

    def save(self, sess, save_path='./model.ckpt'):
        """save model"""
        self.saver.save(sess, save_path=save_path)

    def load(self, sess, save_path='./model.ckpt'):
        """load model"""
        print('try load model from', save_path)
        self.saver.restore(sess, save_path)

if __name__ == '__main__':

    nn = NN()
    nn.train()
