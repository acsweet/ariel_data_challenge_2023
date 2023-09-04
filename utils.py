import tensorflow as tf
import numpy as np
import math
import keras.backend as K
import h5py

def tf_data_noise(input_noise, dtype=tf.dtypes.float64):
    return tf.random.normal(input_noise.shape, mean=0, stddev=input_noise, dtype=dtype)

def normalize_with_noise(input_specta, input_noise, method='-mean', conc=0, glbl_mean=None, glbl_std=None):
    # https://gitlab.astro.rug.nl/ardevol/exocnn/-/blob/main/Code/exobaconn.py

    supported_methods = ['-mean']
    # TODO: make this meaningful
    if method not in supported_methods:
        raise Exception(f"unsupported method: {{{method}}}, supported methods: {supported_methods}")

    inputs_with_noise = tf.math.add(input_specta, tf_data_noise(input_noise))
    
    mean_values =  tf.math.reduce_mean(inputs_with_noise)
    inputs_sub_mean = tf.subtract(inputs_with_noise, mean_values)
    if conc == 1:
        if glbl_mean is not None and glbl_std is not None:
            inputs_with_noise = (inputs_with_noise - glbl_mean)/glbl_std
        return tf.stack([inputs_with_noise, inputs_sub_mean], axis=1)
    else: 
        return tf.stack(inputs_sub_mean)

def sample_trace_data(trace_file, target_scale_array):
    min_vals = target_scale_array[:, 0]
    max_vals = target_scale_array[:, 1]

    def f(key):
        trace_file_id = key.numpy()
        trace = trace_file[trace_file_id]['tracedata']
        weights = trace_file[trace_file_id]['weights']

        trace_id = np.random.choice(np.arange(len(weights)), p=weights) # sample based on weight
        return (trace[trace_id, :] - min_vals)/(max_vals - min_vals)
    
    return f

def get_tf_datasets(spectra, noise, aux_data, targets, repeat_count, batch_size):
    input_data = {
        'spectra_input': spectra, 
        'noise': noise,
        'added_input': aux_data
    }
    output_data = {
        'dense_output': targets
    }

    tf_dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
    tf_dataset = tf_dataset.map(
        lambda inputs, outputs: (
            {
                'spectra_input': normalize_with_noise(inputs['spectra_input'], inputs['noise'], conc=0),
                'added_input': inputs['added_input']
            }, 
            {'dense_output': outputs['dense_output']}
        )
    )
    tf_dataset = tf_dataset.repeat(repeat_count)
    tf_dataset = tf_dataset.batch(batch_size)

    return tf_dataset

def get_tf_dataset_sampled(spectra, noise, aux_data, targets, repeat_count, batch_size, trace_file, target_scales_arr, target_labels=None, split_outputs=False):
    input_data = {
        'spectra_input': spectra, 
        'noise': noise,
        'added_input': aux_data
    }
    output_data = {
        'outputs': targets
    }

    tf_dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
    additional_string = tf.constant('Planet_')
    tf_dataset = tf_dataset.map(
        lambda x, y: (x, {'outputs': tf.strings.join([additional_string, y['outputs']])})
    )
    tf_dataset = tf_dataset.map(
        lambda inputs, outputs: (
            {
                'spectra_input': normalize_with_noise(inputs['spectra_input'], inputs['noise'], conc=0),
                'added_input': inputs['added_input']
            }, 
            tf.py_function(sample_trace_data(trace_file, target_scales_arr), [outputs['outputs']], Tout=tf.float64)
        )
    )
    if split_outputs:
        tf_dataset = tf_dataset.map(
            lambda inputs, outputs: (
                inputs, 
                {f'output_{t}': outputs[i] for i, t in enumerate(target_labels)}
            )
        )
    tf_dataset = tf_dataset.repeat(repeat_count)
    tf_dataset = tf_dataset.batch(batch_size)

    return tf_dataset

def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90,
                                   verbose=1):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=verbose)

    return learning_rate_scheduler

def fetch_callbacks(epochs, batch_size, weights_file, 
                    max_learn_rate=1e-3, 
                    end_learn_rate=1e-5, 
                    warmup_epoch_count=5,
                    patience=5,
                    ignore_lr_sched=False):
    model_checkpoint =  tf.keras.callbacks.ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True,
                                                    save_weights_only=True, mode='auto', verbose=0)

    lr_scheduler = create_learning_rate_scheduler(max_learn_rate=max_learn_rate,
                                            end_learn_rate=end_learn_rate,
                                            warmup_epoch_count=warmup_epoch_count,
                                            total_epoch_count=epochs,
                                            verbose=0)

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    if ignore_lr_sched:
        return [model_checkpoint, early_stopper]
    else:
        return [model_checkpoint, lr_scheduler, early_stopper]
    
def heteroscedastic_loss_wrapper(D, _float_precision='float32'):
    # https://github.com/exoml/plan-net/blob/master/code/plan_net.py
    # https://gitlab.astro.rug.nl/ardevol/exocnn/-/blob/main/Code/exobaconn.py

    # float64 or float32??
    supported_precisions = ['float32', 'float64']
    if _float_precision not in supported_precisions:
        raise ValueError('{} precision not supported in {}, supported: {}'.format(_float_precision, 
                                                                                  'heteroscedastic_loss_wrapper', 
                                                                                  supported_precisions))
    _dtype = tf.dtypes.float32 if _float_precision == 'float32' else tf.dtypes.float64

    def heteroscedastic_loss(true, pred):
        mean = pred[:, :D]
        L = pred[:, D:]
        N = tf.shape(true)[0]
        k = 1
        inc = 0
        Z = []
        diag = []
        for d in range(D):
            if k == 1:
                Z.append(tf.concat([tf.exp(tf.reshape(L[:,inc:inc+k],[N,k])), tf.zeros((N,D-k), dtype=_dtype)],1))
            else:
                Z.append(tf.concat([tf.reshape(L[:,inc:inc+k-1],[N,k-1]), 
                                    tf.exp(tf.reshape(L[:,inc+k-1],[N,1])), 
                                    tf.zeros((N,D-k), dtype=_dtype)], 1))
            diag.append(K.exp(L[:,inc+k-1]))
            inc += k
            k+=1
        diag = tf.cast(tf.concat(tf.expand_dims(diag,-1),-1), dtype=_dtype)
        lower = tf.reshape(tf.concat(Z,-1),[N,D,D])
        S_inv = tf.matmul(lower,tf.transpose(lower,perm=[0,2,1]))
        x = tf.expand_dims((true - mean),-1)
        quad = tf.matmul(tf.matmul(tf.transpose(x,perm=[0,2,1]), S_inv), x)
        log_det = - 2 * tf.cast(K.sum(K.log(diag), 0), dtype=_dtype)
        return K.mean(tf.squeeze(quad,-1) + log_det, 0)
    
    return heteroscedastic_loss

def quantile_loss_wrapper(perc, delta=1e-4):
    # https://www.kaggle.com/code/abiolatti/deep-quantile-regression-in-keras?scriptVersionId=38395930&cellId=6

    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = tf.cast(perc.reshape(1, -1), tf.float64)
    delta = tf.cast(delta, tf.float64)
    
    def _qloss(y, pred):
        y = tf.expand_dims(y, -1)
        I = tf.cast(y <= pred, tf.float64)
        d = tf.cast(tf.math.abs(y - pred), tf.float64)
        correction = I * (1 - perc) + (1 - I) * perc
        
        huber_loss = tf.math.reduce_sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), axis=1)

        # quantile ordering loss for regularization
        q_order_loss = tf.math.reduce_sum(tf.math.maximum(0.0, pred[:, :-1] - pred[:, 1:] + 1e-6), axis=1)
        return huber_loss + tf.cast(q_order_loss, tf.float64)
    return _qloss

def get_diag(pred, num_dim, num_rows):
    # https://github.com/exoml/plan-net/blob/master/code/plan_net.py
    # https://gitlab.astro.rug.nl/ardevol/exocnn/-/blob/main/Code/exobaconn.py
    
    _dtype = tf.dtypes.float32 # if _float_precision == 'float32' else tf.dtypes.float64
    
    D = num_dim
    mean = pred[:, :D]
    L = pred[:, D:]
    N = num_rows
    k = 1
    inc = 0
    Z = []
    diag = []
    for d in range(D):
        if k == 1:
            Z.append(tf.concat([tf.exp(tf.reshape(L[:,inc:inc+k],[N,k])), tf.zeros((N,D-k), dtype=_dtype)],1))
        else:
            Z.append(tf.concat([tf.reshape(L[:,inc:inc+k-1],[N,k-1]), 
                                tf.exp(tf.reshape(L[:,inc+k-1],[N,1])), 
                                tf.zeros((N,D-k), dtype=_dtype)], 1))
        diag.append(K.exp(L[:,inc+k-1]))
        inc += k
        k+=1
    diag = tf.cast(tf.concat(tf.expand_dims(diag,-1),-1), dtype=_dtype)
    lower = tf.reshape(tf.concat(Z,-1),[N,D,D])
    return lower, diag

def ariel_resolution():
    import numpy as np
    from taurex.util.util import wnwidth_to_wlwidth

    wlgrid = np.array([0.55      , 0.7       , 0.95      , 1.156375  , 1.27490344,
       1.40558104, 1.5496531 , 1.70849254, 1.88361302, 1.9695975 ,
       2.00918641, 2.04957106, 2.09076743, 2.13279186, 2.17566098,
       2.21939176, 2.26400154, 2.30950797, 2.35592908, 2.40328325,
       2.45158925, 2.50086619, 2.5511336 , 2.60241139, 2.65471985,
       2.70807972, 2.76251213, 2.81803862, 2.8746812 , 2.93246229,
       2.99140478, 3.05153202, 3.11286781, 3.17543645, 3.23926272,
       3.30437191, 3.37078978, 3.43854266, 3.50765736, 3.57816128,
       3.65008232, 3.72344897, 4.03216667, 4.30545796, 4.59727234,
       4.90886524, 5.24157722, 5.59683967, 5.97618103, 6.3812333 ,
       6.81373911, 7.2755592 ])[::-1]
    wlwidth = np.array([0.10083333, 0.20416667, 0.30767045, 0.11301861, 0.12460302,
       0.13737483, 0.15145575, 0.16697996, 0.18409541, 0.03919888,
       0.03998678, 0.04079051, 0.0416104 , 0.04244677, 0.04329995,
       0.04417028, 0.0450581 , 0.04596377, 0.04688764, 0.04783008,
       0.04879147, 0.04977218, 0.0507726 , 0.05179313, 0.05283417,
       0.05389614, 0.05497945, 0.05608453, 0.05721183, 0.05836179,
       0.05953486, 0.06073151, 0.06195222, 0.06319746, 0.06446773,
       0.06576353, 0.06708537, 0.06843379, 0.06980931, 0.07121248,
       0.07264385, 0.07410399, 0.26461764, 0.28255283, 0.30170364,
       0.32215244, 0.34398722, 0.36730191, 0.39219681, 0.41877904,
       0.44716295, 0.47747067])[::-1]
    wngrid = 10000/wlgrid
    wnwidth = wnwidth_to_wlwidth(wlgrid,wlwidth )
    return wlgrid, wlwidth, wngrid, wnwidth