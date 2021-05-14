import numpy as np
import tensorflow as tf

def make_CBFNet(params):
    ''' 
    Make a standard feed-forward CNN. 
    params: dict with 
                'num_conv_layers', 'input_shape', 'dropout', 'dropout_mc',
                'conv-in_featN', 'conv_featN' 
    '''

    class MonteCarloDropout(tf.keras.layers.Dropout):

        def __init__(self, rate=0.5):
            super(MonteCarloDropout, self).__init__(rate)
            self.rate = rate

        def call(self, inputs):
            return super().call(inputs, training=True)

    # Define model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(params['conv-in_featN'],3,
                          padding='same',
                          activation='relu',
                          input_shape=params['input_shape']))

    for _ in range(params['num_conv_layers']):
        model.add(tf.keras.layers.Conv3D(params['conv_featN'],
                          3, padding='same',
                          activation='relu',
                          kernel_initializer='glorot_normal'))
        if params['dropout_mc']:
            model.add(MonteCarloDropout(params['dropout']))
        else:
            model.add(tf.keras.layers.Dropout(rate=params['dropout']))

    model.add(tf.keras.layers.Conv3D(10, 3, padding='same',
                          activation='relu'))
    if params['dropout_mc']:
        model.add(MonteCarloDropout(rate=params['dropout']))
    else:
        model.add(tf.keras.layers.Dropout(rate=params['dropout']))
    model.add(tf.keras.layers.Conv3D(2, 2, padding='same',
                          activation='relu'))

    return model

def downsample(filters, size, strides, apply_batchnorm=True, apply_dropout=False, dropout_mc=False):
  '''
  A single downsample convolutional layer with optional batchnorm, 
  dropout, and test-time dropout.
  '''

  class MonteCarloDropout(tf.keras.layers.Dropout):

        def __init__(self, rate=0.5):
            super(MonteCarloDropout, self).__init__(rate)
            self.rate = rate

        def call(self, inputs):
            return super().call(inputs, training=True)

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      if dropout_mc:
          result.add(MonteCarloDropout(rate=0.3))
      else:
          result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, strides, apply_dropout=False, dropout_mc=False):
  '''
  A single upsample convolutional layer with optional dropout,
  and test-time dropout.
  '''
  class MonteCarloDropout(tf.keras.layers.Dropout):

        def __init__(self, rate=0.5):
            super(MonteCarloDropout, self).__init__(rate)
            self.rate = rate

        def call(self, inputs):
            return super().call(inputs, training=True)

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv3DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      if dropout_mc:
          result.add(MonteCarloDropout(rate=0.3))
      else:
          result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.ReLU())

  return result

def make_uCBFNet(params):
    '''
    Make a U-Net model using the above downsample and upsample functions.
    
    params: dict with input params. Example:
        {"input_shape": input_shape,
        "conv-n_in": 10,
        "conv-n_deep": 60,
        "loss_brain": 0.5,
        "lr": 0.0005,
        "dropout_mc": False}
    '''

    input_size = params['input_shape']
    dropout_mc = params['dropout_mc']

    inputs = tf.keras.layers.Input(input_size)
    
    down_stack = [
        downsample(params['conv-n_in'], 3, (2,2,1), apply_batchnorm=False),
        downsample(20, 3, (2,2,1)),
        downsample(params['conv-n_deep'], 3, (2,2,1), apply_dropout=True, dropout_mc=dropout_mc)
    ]
    
    up_stack = [
        upsample(20, 3, (2,2,1), apply_dropout=True, dropout_mc=dropout_mc),
        upsample(10, 3, (2,2,1), apply_dropout=True, dropout_mc=dropout_mc)
    ]
    
    last_stack = [
        tf.keras.layers.Conv3DTranspose(10, 3, strides=(2,2,1), padding='same', activation='relu'),
        tf.keras.layers.Conv3D(2, 3, padding='same', activation='relu')]
    x = inputs
    
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x,skip])
    
    for layer in last_stack: x = layer(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def masked_loss(mask, weights, loss_fn='mae', both=True):
    '''
    Given a brain mask, compute a loss with given weighting between
    brain and background voxels.

    mask: 3D numpy array giving the desired mask
    weights: Two-element list with relative weights for brain 
            and background voxels
    loss_fn: String specifying either mae or huber loss function
    both: Extend the mask into a 4D mask with size 2 in axis -1
    '''

    if both:
        mask = np.repeat(mask[None,:,:,:,None], 2, axis=-1)
        
    def loss(y_true, y_pred):
        masked_true = y_true[mask>0]
        masked_pred = y_pred[mask>0]
        bg_true = y_true[mask<1]
        bg_pred = y_pred[mask<1]
        if loss_fn == 'huber':
            l = tf.losses.Huber()
        elif loss_fn == 'mae':
            l = tf.losses.MeanAbsoluteError()
        mask_loss = l(masked_true, masked_pred)
        bg_loss = l(bg_true, bg_pred)
        return 100*(weights[0]*mask_loss+weights[1]*bg_loss)
    return loss