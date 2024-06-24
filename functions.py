# License
#  This program is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published 
#  by the Free Software Foundation, either version 3 of the License, 
#  or (at your option) any later version.
#  This program is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#  See the GNU General Public License for more details. You should have 
#  received a copy of the GNU General Public License along with this 
#  program. If not, see <https://www.gnu.org/licenses/>. 

# Description
# This file contains the functions that are used in the main code

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pickle
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import SimpleRNN, Dense, GRU, Input
from sklearn.preprocessing import MinMaxScaler
from distutils.dir_util import mkpath
from tqdm import tqdm
import input_file
from input_file import *
import copy
      

def create_datasets(sequence_lenght, number_strain_sequences, 
                    strains_path, simo_results_path, splitter):
    
    total_indices = range(0, number_strain_sequences)
    training_indices = random.sample(total_indices, 
                                     round(splitter[0]* \
                                           number_strain_sequences))
    possible_validation_indices = list(set(total_indices) - \
                                       set(training_indices))
    validation_indices = random.sample(possible_validation_indices, 
                                          round(splitter[1]\
                                        *  number_strain_sequences))
    test_indices = list(set(total_indices) - set(training_indices) - \
                        set(validation_indices))
        
    x_train = np.zeros([len(training_indices), 
                            sequence_lenght, 1])
    y_train = np.zeros([len(training_indices), 
                            sequence_lenght, 1])
    x_validation = np.zeros([len(validation_indices), 
                            sequence_lenght, 1])
    y_validation = np.zeros([len(validation_indices), 
                            sequence_lenght, 1])
    x_test = np.zeros([len(test_indices), 
                            sequence_lenght, 1])
    y_test = np.zeros([len(test_indices), 
                            sequence_lenght, 1])
    
    print("Creating training set")
    for i in tqdm(range(0, len(training_indices))):
        x_train[i,:,:] = (np.loadtxt(strains_path + 
                         '/%i.txt' %(training_indices[i]))[:, 1])[:, 
                                                                    np.newaxis]
        y_train[i,:,:] = (np.loadtxt(simo_results_path + 
                         '/%i.txt' %(training_indices[i]))[:, 1])[:, 
                                                                    np.newaxis]

    print("Creating validation set")
    for i in tqdm(range(0, len(validation_indices))):
        x_validation[i,:,:] = (np.loadtxt(strains_path + 
                               '/%i.txt' %(validation_indices[i]))[:, 1])[:, 
                                                                    np.newaxis]
        y_validation[i,:,:] = (np.loadtxt(simo_results_path + 
                               '/%i.txt' %(validation_indices[i]))[:, 1])[:, 
                                                                    np.newaxis]

    print("Creating test set")
    for i in tqdm(range(0, len(test_indices))):
        x_test[i,:,:] = (np.loadtxt(strains_path + 
                         '/%i.txt' %(test_indices[i]))[:, 1])[:, np.newaxis]
        y_test[i,:,:] = (np.loadtxt(simo_results_path + 
                         '/%i.txt' %(test_indices[i]))[:, 1])[:, np.newaxis]

    return x_train, y_train, x_validation, y_validation, x_test, y_test
             

def serialise_dataset(x_train, y_train, x_validation, y_validation, x_test, 
                      y_test, y_prediction, route):
    with open(route + '/x_training.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    f.close()
    with open(route + '/y_training.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    f.close()
    with open(route + '/x_validation.pkl', 'wb') as f:
        pickle.dump(x_validation, f)
    f.close()
    with open(route + '/y_validation.pkl', 'wb') as f:
        pickle.dump(y_validation, f)
    f.close()
    with open(route + '/x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    f.close()
    with open(route + '/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    f.close()
    with open(route + '/y_prediction.pkl', 'wb') as f:
        pickle.dump(y_prediction, f)
    f.close()

def create_scaled_dataset(x_scaler, y_scaler, x_train, y_train, x_validation, 
                          y_validation, x_test, y_test):
    
    x_scaler.fit(x_train.reshape([x_train.shape[0] * x_train.shape[1], 
                                     x_train.shape[2]]))

    y_scaler.fit(y_train.reshape([y_train.shape[0] * y_train.shape[1], 
                                     y_train.shape[2]]))

    
    x_train_scaled = x_scaler.transform(x_train.reshape([x_train.shape[0] *
                                                         x_train.shape[1], 
                                                         x_train.shape[2]]))
    x_train_scaled = x_train_scaled.reshape(x_train.shape)

    y_train_scaled = y_scaler.transform(y_train.reshape([y_train.shape[0] *
                                                         y_train.shape[1], 
                                                         y_train.shape[2]]))

    y_train_scaled = y_train_scaled.reshape(y_train.shape)

    x_validation_scaled = x_scaler.transform(x_validation.reshape([
                                                       x_validation.shape[0] *
                                                       x_validation.shape[1], 
                                                       x_validation.shape[2]]))
    
    x_validation_scaled = x_validation_scaled.reshape(x_validation.shape)

    y_validation_scaled = y_scaler.transform(y_validation.reshape([
                                                        y_validation.shape[0] *
                                                         y_validation.shape[1], 
                                                       y_validation.shape[2]]))

    y_validation_scaled = y_validation_scaled.reshape(y_validation.shape)


    x_test_scaled = x_scaler.transform(x_test.reshape([x_test.shape[0] *
                                                       x_test.shape[1], 
                                                       x_test.shape[2]]))
    x_test_scaled = x_test_scaled.reshape(x_test.shape)

    y_test_scaled = y_scaler.transform(y_test.reshape([y_test.shape[0] *
                                                       y_test.shape[1], 
                                                       y_test.shape[2]]))
    y_test_scaled = y_test_scaled.reshape(y_test.shape)


    return x_train_scaled, y_train_scaled, x_validation_scaled, \
           y_validation_scaled, x_test_scaled, y_test_scaled


def plot_results(x, y_test, y_pred, plots_path, history,
             number_of_plots = 30):
    places=np.arange(number_of_plots)
    mkpath(plots_path)
    for i in places:
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(111) #
        coord_x = x[i,:,0]
        y_test_coords = y_test[i,:,0]
        y_pred_coords = y_pred[i,:,0]
        plot_name  = plots_path + '/test_sample_number_' + str(i) \
                                     + '.png'
        plt.plot(np.cumsum(coord_x), y_test_coords, label = 'Expected', 
                 linewidth=3, color = 'green', marker = "x", alpha = 0.3)
        plt.plot(np.cumsum(coord_x), y_pred_coords, label = 'Predicted',
                 color = 'black', linestyle='--')
        box = ax.get_position() #
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])#
        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Strain (mm/mm)", fontsize = 16)
        plt.ylabel("Stress (GPa)", fontsize = 16)
        plt.grid(True)
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()
    plt.figure()
    pd.DataFrame(history.history).plot(figsize=(15, 10))
    plot_name = plots_path + '/Convergence history.png'
    plt.grid(True)
    # plt.legend(fontsize=15)
    plt.yscale("log")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Loss", fontsize = 16)
    plt.savefig(plot_name, bbox_inches='tight')

def calculate_stresses_for_all(strain_path, stress_folder_path):
    print ("Calculating Simo's stresses")
    mkpath(stress_folder_path)
    for i in tqdm(range(0, NUMBER_STRAIN_SEQUENCES)):
        with open(strain_path+'/%i.txt' %(i), 'r') as f:
            strains = np.loadtxt(strain_path +'/%i.txt' %(i))[:,1]
            dummy_array = np.array([])
            eps_results, sigma_results  =  \
            calculate_stress_per_strain_sequence(copy.deepcopy(K),
                                                 copy.deepcopy(eps_current),
                                                 copy.deepcopy(eps_p_current),
                                                 copy.deepcopy(alpha_current),
                                                 copy.deepcopy(E),
                                                 copy.deepcopy(sigma_current), 
                                                 copy.deepcopy(SIGMA_Y),
                                                 copy.deepcopy(strains))
            eps_results = eps_results[:,np.newaxis]
            sigma_results = sigma_results[:,np.newaxis]
            dummy_array = np.concatenate((eps_results,
                                          sigma_results),axis = 1)
            np.savetxt('./' + stress_folder_path + '/'
                       + '%i.txt' %(i), dummy_array,delimiter = ' ')
    print("Simo's calculation finished")


def create_simple_rnn(nodes):
    model = Sequential()
    model.add(GRU(nodes, return_sequences = True,
                       input_shape = [None, 1])),
    model.add(Dense(units = nodes, kernel_initializer = 'he_normal',
                         activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'he_normal',
                         activation = 'linear'))

    return model


def create_simple_cnn(nodes):
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for rate in (1, 2, 4, 8) * 2:
        model.add(keras.layers.Conv1D(filters = nodes, kernel_size = 2, 
                                           padding="causal", activation = \
                                           "relu", dilation_rate=rate))
            
    model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
    return model



def create_encoder_decoder(nodes):
    learning_rate = 0.01
    decay = 0 # Learning rate decay
    optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) 
    loss = "mse" 
               
    # #encoder
    encoder_inputs = Input(shape=(None, 1))
    encoder = GRU(nodes, return_state=True)
    _, encoder_states = encoder(encoder_inputs)
    
    # decoder
    decoder_inputs = Input(shape=(None, 1))
    decoder = GRU(nodes, return_state = True, return_sequences=True)
    decoder_output, _ = decoder(decoder_inputs, initial_state = 
                                encoder_states)
    
    decoder_dense_1 = Dense(units = nodes, kernel_initializer = \
                            'he_normal', activation = 'relu')(decoder_output)
    output = Dense(units = 1, kernel_initializer = 'he_normal',
                                      activation = 'linear')(decoder_dense_1)           
    model = Model(inputs = [encoder_inputs, decoder_inputs], 
                  outputs = output)
    model.compile(optimizer = optimiser, loss=loss)
    model = model
    return model
            
def train_enc_dec(model, x_train, y_train, x_validation, y_validation, x_test, 
             y_test, number_of_epochs):
        # if (tunable == False):
        # print("shape of x train:")
        # print(x_train.shape)

        decoder_input_data = np.zeros((x_train.shape))
        decoder_validation_input_data = np.zeros((x_validation.shape))
        # print("shape of decoder input data:")
        # print(decoder_input_data.shape)
        
        # print("shape of y train input data:")
        # print(y_train.shape)
        
        history  =  model.fit([x_train, decoder_input_data],
                              y_train, epochs = number_of_epochs, 
                              validation_data = ([x_validation, 
                                            decoder_validation_input_data],
                                            y_validation))
        # history = history
        # model.summary()
        return history
            
def test_enc_dec(model, x_train, y_train, x_validation, y_validation, x_test, 
             y_test, number_of_epochs):
    # print(" Result from test")
    # print("shape of x test:")
    # print(x_test.shape)

    test_decoder_input_data = np.zeros((x_test.shape))
    # print("shape of test_decoder_input_data:")
    # print(test_decoder_input_data.shape)

    # print("shape of y test input data:")
    # print(y_test.shape)

    testResults = model.evaluate([x_test, test_decoder_input_data], y_test)
    # print("testing Finished")
        
def predict_enc_dec(model, Xnew = [0]):
    decoder_xnew = np.zeros((Xnew.shape))
    return model.predict([Xnew, decoder_xnew])


def create_conv_rec_nn(nodes):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for rate in (1, 2, 4, 8) * 2:
        model.add(keras.layers.Conv1D(filters=nodes, kernel_size=2, 
                                           padding="causal",
                                           activation="relu", 
                                           dilation_rate=rate))
    model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
    
    # model.add(GRU(nodes, return_sequences=True)),
    model.add(GRU(10, return_sequences=True)),    
    # model.add(Dense(units = nodes, kernel_initializer = 'he_normal',
    #                 activation = 'relu'))
    model.add(Dense(units = 10, kernel_initializer = 'he_normal',
                    activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'he_normal',
                    activation = 'linear'))
    return model

def compile_nn(model, opt, loss_function):
    model.compile(optimizer=opt, loss= loss_function)

def train_nn(model, x_train, y_train, x_validation, y_validation, x_test, 
             y_test, number_of_epochs):
    history = model.fit(x_train, y_train,  epochs = number_of_epochs,
                              validation_data = (x_validation,  y_validation))
    return history


def calculate_stress_per_strain_sequence(K , eps_current, eps_p_current, 
                                         alpha_current, E, sigma_current, 
                                         sigma_y, strains):
    sigma_results = []
    eps_results = []
    
    for delta_strain in strains:
        Eps_new = eps_current+delta_strain
        Sigma_trial_new = E*(Eps_new-eps_p_current)
        f_trial_new = abs(Sigma_trial_new)-(sigma_y+K * alpha_current)
        if (f_trial_new <= 0):
            Sigma_new = Sigma_trial_new
        else:
            DeltaGamma = f_trial_new/(E+K)
            Sigma_new = (1-(DeltaGamma*E)/(abs(Sigma_trial_new)))\
                        *Sigma_trial_new
            Eps_P_new = eps_p_current+DeltaGamma \
                        *np.sign(Sigma_trial_new)
            alpha_new = alpha_current+DeltaGamma
            eps_p_current = Eps_P_new
            alpha_current = alpha_new
        eps_current = Eps_new
        sigma_results.append(Sigma_new)
        eps_results.append(Eps_new)
    return np.array(eps_results), np.array(sigma_results)
    
        
def writestrains(strains_path, max_deformation, 
                 number_random_points, number_interpolation_points):
    print ("Generating strains (paths)")
    # mkpath(parent_folder_path)
    mkpath(strains_path)

    for i in tqdm(range(0, NUMBER_STRAIN_SEQUENCES)):
        #Sample from a standard normal distribution
        mu, sigma = 0, 1 #mean and variance
        random_points = np.zeros(number_random_points+1)
        random_points[1:] = np.random.normal(mu, sigma, number_random_points)
        final_path = interpolate_control_points(random_points, 
                                                number_interpolation_points)
        max_def = np.max(abs(np.cumsum(final_path)))
        if(max_def > max_deformation):
            final_path = final_path * max_deformation/max_def       
        dummy=np.zeros((len(final_path),2))
        dummy[:,0] = range(len(final_path))
        dummy[:,1] = final_path
        np.savetxt('./' + strains_path + '/' + '%i.txt' %(i), dummy, \
                   delimiter = ' ')
            
def write_random_strains(strains_path, max_deformation, 
                 number_random_points, number_interpolation_points):
    print ("Generating strains (paths)")
    mkpath(strains_path)

    for i in tqdm(range(0, NUMBER_STRAIN_SEQUENCES)):
        #Sample from a standard normal distribution
        mu, sigma = 0, 1 #mean and variance
        random_points = np.zeros(number_random_points * 
                                 number_interpolation_points + 1)
        random_points[1:] = np.random.normal(mu, sigma, number_random_points * 
                                 number_interpolation_points)
        final_path = np.copy(random_points) 
        max_def = np.max(abs(np.cumsum(final_path)))
        if(max_def > max_deformation):
            final_path = final_path * max_deformation/max_def       
            max_def = np.max(abs(np.cumsum(final_path)))
            # print(max_def)
        dummy=np.zeros((len(final_path),2))
        dummy[:,0] = range(len(final_path))
        dummy[:,1] = final_path
        np.savetxt('./' + strains_path + '/' + '%i.txt' %(i), dummy, \
                   delimiter = ' ')
            

def interpolate_control_points(random_points, number_interpolation_points):
    diff = np.diff(random_points)
    interpolated = np.zeros((len(random_points)-1) * 
                            number_interpolation_points+1)
    for i in range(len(random_points)-1):
        dummy=(random_points[i+1]-random_points[i]) / \
        (number_interpolation_points-1)
        interpolated[i * number_interpolation_points + 1:(i + 1) * \
                     (number_interpolation_points)+1]=dummy
    return interpolated


def plot_acc_strain(plots_path, x_train, number_sequences):
    mkpath(plots_path)
    for i in range(number_sequences):
        plt.figure(figsize=(15, 10))
        plt.plot(range(x_train.shape[1]-1), np.cumsum(x_train[i, 1:]), 
                 linewidth=3)
        plot_name  = plots_path + "sequence_" + str(i) + '.png'
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("time step", fontsize = 20)
        plt.ylabel("Accumulated strain (mm/mm)", fontsize = 20)
        plt.grid(True)
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()