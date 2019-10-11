import tensorflow as tf
import tensorflow.keras as keras
from lib.base_proc import corr_mat

class Tf_logisticRegression:
    def __init__(self):
        self.coeff = None
        self.feature_importance = None
        self.L = None
        self.gini = None
        self.model_ = None
        self.cm = None

    def corr_check(self, data, x1, x2, method='pearson'):
        if self.cm is None:
            self.cm = corr_mat(data, self.col_features, method=method)
        return self.cm.loc[x1, x2]


    def create_model(self,
                     n_features,
                     kernel_penalty=None,
                     kernel_L1=None,
                     kernel_L2=None,
                     bias_penalty=None,
                     bias_L1=None,
                     bias_L2=None,
                     use_bias=True,
                     optimizer='Adam',
                     adam_lr=1e-3,
                     SGD_lr=0.01,
                     SGD_momentum=0.0
                     ):

        # Kernel regulizer
        if kernel_penalty == 'L1':
            kernel_regulizer = tf.keras.regularizers.l1(l1=kernel_L1)
        elif kernel_penalty == 'L2':
            kernel_regulizer = tf.keras.regularizers.l2(l2=kernel_L2)
        elif kernel_penalty == 'elastic':
            kernel_regulizer = tf.keras.regularizers.l1_l2(l1=kernel_L1, l2=kernel_L2)
        else:
            kernel_regulizer = None

        # Bias Regulizer:
        if bias_penalty == 'L1':
            bias_regulizer = tf.keras.regularizers.l1(l1=bias_L1)
        elif bias_penalty == 'L2':
            bias_regulizer = tf.keras.regularizers.l2(l2=bias_L2)
        elif bias_penalty == 'elastic':
            bias_regulizer = tf.keras.regularizers.l1_l2(l1=bias_L1, l2=bias_L2)
        else:
            bias_regulizer = None

        # Optimizer
        if optimizer == 'Adam': optimizer = keras.optimizers.Adam(lr=adam_lr)
        elif optimizer == 'SGD': optimizer = keras.optimizers.SGD(learning_rate=SGD_lr,
                                                                  momentum=SGD_momentum)

        inputs = keras.Input(shape=(n_features,))
        outputs = keras.Dense(1,
                              activation=tf.nn.sigmoid,
                              kernel_regulazier=kernel_regulizer,
                              bias_regulizer=bias_regulizer,
                              use_bias=use_bias)(inputs)

        self.model_ = keras.Model(inputs, outputs)

        self.model_.compile(
                            optimizer=optimizer,
                            loss=keras.losses.BinaryCrossentropy(),
                            metrics=['auc'])

