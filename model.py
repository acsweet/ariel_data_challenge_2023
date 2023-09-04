from keras.layers import (
    Dense, Input, Concatenate, BatchNormalization, Dropout, Flatten, ReLU
)
from keras.models import Model
import tensorflow as tf
from tabnet.custom_objects import GroupNormalization

def residual_block(X, kernels, kernel_size, res_block_num=None):
    # this is to get x aligned with subsequent layers, along the "channel" axis
    X = tf.keras.layers.Conv1D(kernels, kernel_size, padding='same', name='pre_res_conv_{}'.format(res_block_num))(X)
    X = GroupNormalization(axis=-1, groups=1)(X)
    X = tf.keras.layers.ReLU()(X)

    out = tf.keras.layers.Conv1D(kernels, kernel_size, padding='same', name='rb_{}_conv_1'.format(res_block_num))(X)
    out = GroupNormalization(axis=-1, groups=1)(out)
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Conv1D(kernels, kernel_size, padding='same', name='rb_{}_conv_2'.format(res_block_num))(out)
    out = GroupNormalization(axis=-1, groups=1)(out)
    out = tf.keras.layers.add([X, out])
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.MaxPool1D()(out)
    return out

def get_model(loss, num_features, n_feature_cols, num_added_features, num_targets, target_labels=None,
                        quantiles=None, weights_file=None, kernels = [16, 32, 64, 128], 
                        kernel_size = 3, dropout_training = False, dropout_p = 0.1, 
                        act_mu = 'sigmoid', act_cov = 'linear', act_d='relu', expect_partial=True):

    added_feature_input_layer = Input(shape=(num_added_features,), name='added_input')
    a_x = Dense(500, name='aux_dense_1')(added_feature_input_layer)
    a_x = ReLU()(a_x)

    if n_feature_cols is None:
        spectra_input_layer = Input(shape=(num_features, ), name='spectra_input')
        x = tf.expand_dims(spectra_input_layer, -1)
    else:
        spectra_input_layer = Input(shape=(num_features, n_feature_cols), name='spectra_input')
        x = spectra_input_layer

    for j, k in enumerate(kernels):
        x = residual_block(x, k, kernel_size, res_block_num=j)

    x = Flatten()(x)

    x = Concatenate(axis=-1)([x, a_x])
    x = Dense(500, activation=act_d, name='concat_dense_1')(x)
    x = Dropout(dropout_p)(x, training=dropout_training)
    x = Dense(100, activation=act_d, name='concat_dense_2')(x)
    x = Dropout(dropout_p, name='final_dropout')(x, training=dropout_training)

    if loss == 'mse':
        outputs = Dense(num_targets, activation=None, name='dense_output')(x)
    elif loss =='chol':
        mu = Dense((num_targets), activation=act_mu, name='dense_mu')(x)
        covariance = Dense(num_targets*(num_targets+1)/2, activation=act_cov, name='dense_cov')(x)
        outputs = tf.keras.layers.concatenate([mu, covariance])
    elif loss == 'uniform_quantiles':
        outputs = []
        for i in range(len(target_labels)):
            outputs.append(Dense(len(quantiles), activation=None, name='output_{}'.format(target_labels[i]))(x))

    model = Model(inputs=[spectra_input_layer, added_feature_input_layer], outputs=outputs)
    
    if weights_file is not None:
        if expect_partial:
            model.load_weights(weights_file).expect_partial()
        else:
            model.load_weights(weights_file)

    return model