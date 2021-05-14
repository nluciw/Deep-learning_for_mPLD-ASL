''' This is the main script for training CBFnet and saving the model +
trained parameters.

The script can perform a simple training on full data inputs or curriculum
training on inputs deprived of channels.

You can type python main.py -h for help with the input options.'''

import os
import tensorflow as tf
import argparse
from load_data import organize_filenames, get_dataset, numpy_from_nifti
from models import make_uCBFNet, masked_loss
from training import train_basic_model
import pickle

# Get input arguments. Can also add additional lines for parameters like project_dir
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, help='Number of training epochs')
parser.add_argument("-o", "--out", help='Name of output directory')
parser.add_argument("-c", "--curriculum", type=bool, 
                    help='Boolean indicating whether to do curriculum training after training for e epochs')

args = parser.parse_args()

# Define working directories
project_dir = '/home/nluciw/home/projects/cbfnet/'
data_dir = project_dir + 'data/'
output_dir = project_dir + 'outputs/' + args.out

# Files with train and test lists
train_file = data_dir + 'study_datasets/test/'+'ge_train.txt'
test_file = data_dir + 'study_datasets/test/'+'ge_test.txt'

# Separate files into inputs and outputs
Xtrain_files, ytrain_files = organize_filenames(train_file)
Xtest_files, ytest_files = organize_filenames(test_file)

# Load into TF datasets, crop images if desired
crop = [0,0,0]
train_ds = get_dataset(Xtrain_files, ytrain_files, batch_size=5, crop_size=crop,
                       moments=data_dir+'study_datasets/cbfnet_data/moments.txt')
test_ds = get_dataset(Xtest_files, ytest_files, batch_size=5, crop_size=crop,
                       moments=data_dir+'study_datasets/cbfnet_data/moments.txt')


input_shape = [item[0][0].numpy().shape[-4:] for item in train_ds.take(1)][0]

# Some important paramaters. I've defined them in this way in case we want to read
# params in from a file. Can replace this dict with a json read.
params = {"input_shape": input_shape,
        "conv-n_in": 10,
        "conv-n_deep": 60,
        "loss_brain": 0.5,
        "lr": 0.0005,
        "dropout_mc": False}

# Construct the U-Net model
unet = make_uCBFNet(params)

print(unet.summary())

# Define the optimizer and learning rate
optimizer = tf.keras.optimizers.Adam(params['lr'])

# Get brain mask
mask = numpy_from_nifti([data_dir+'study_datasets/cbfnet_data/mask.nii.gz',],crop=crop)[0,...]
# Define masked loss (ie loss function weighted towards brain & away from background)
mask_loss = masked_loss(mask, [params['loss_brain'], 1-params['loss_brain']], loss_fn='mae')

# Get a brain-only loss for metric tracking
def brain_loss():
    return masked_loss(mask, [1., 0.], loss_fn='mae')
brain_l = brain_loss()

# Configure the model for training
unet.compile(optimizer=optimizer, loss=mask_loss,
             metrics=['mae', tf.losses.Huber(), brain_l])
                                 
epochs = args.epochs

# Train the U-Net
history = train_basic_model(unet, {'train': train_ds, 'val': test_ds},
                            epochs, params, min_delta=0.0005, lr_decay=True)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model + weights
json_config = unet.to_json()

with open(output_dir + '/ge_model_mask0.json', 'w') as json_file:
    json_file.write(json_config)
# Save weights to disk
unet.save_weights(output_dir+'/ge_weights_mask0.h5')

with open(output_dir + '/ge_history_mask0.json', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# Each element identifies the number of removed PLDs in the training inputs.
# E.g. for 1st element the training set has some examples with all PLDs present
# and some with one PLD removed. 
train_schedule = [[0,1], [1,2], [2,3], [3]]

# Do curriculum training for 100 epochs for each element of train_schedule
if args.curriculum:
    for i in range(0,4):

        train_ds = get_dataset(Xtrain_files, ytrain_files, batch_size=5, crop_size=crop,
                           moments=data_dir+'study_datasets/cbfnet_data/moments.txt',
                           mask_num=train_schedule[i],
                           include_all_masked=True,
                           shuffle=True)

        test_ds = get_dataset(Xtest_files, ytest_files, batch_size=5, crop_size=crop,
                           moments=data_dir+'study_datasets/cbfnet_data/moments.txt',
                           mask_num=train_schedule[i],
                           include_all_masked=True,
                           shuffle=True)

        input_shape = [(len(item), len(item[0]), ) for item in train_ds.take(1)]
        print(input_shape)
        if i==3:
            min_delta=0.0002
        else:
            min_delta=0.0005

        history = train_basic_model(unet, {'train': train_ds, 'val': test_ds},
                                    100, params, min_delta)

        json_config = unet.to_json()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        i += 1
        with open(output_dir + '/ge_model_mask{0}.json'.format(i), 'w') as json_file:
            json_file.write(json_config)
        # Save weights to disk
        unet.save_weights(output_dir+'/ge_weights_mask{0}.h5'.format(i))

        with open(output_dir + '/ge_history_mask{0}.json'.format(i), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

