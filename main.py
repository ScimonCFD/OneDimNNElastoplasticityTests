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
#  This code evaluates the performance of various neural networks as surrogate 
#  models for the isotropic hardening constitutive law. The code accomplishes 
#  this through the following main steps:
#  1. Generation of Several Strain Sequences:
#  Multiple strain sequences are generated, with a maximum accumulated strain 
#  restricted to 5%.
#  2. Calculation of Isotropic Hardening Stresses:
#  The code calculates the isotropic hardening stresses corresponding to the 
#  strain sequences using Simo's algorithm.
#  3. Dataset Splitting:
#  The dataset is divided into training (70%), validation (20%), and test (10%) 
#  sets.
#  4. Testing Various Neural Network Designs:
#  Different types of neural network designs are tested on the strain-stress 
#  sequences in the training set. These designs include Convolutional, 
#  Recurrent, Encoder-Decoder, and a combined Convolutional + Recurrent neural 
#  network.
#  5. Plotting Expected vs. Neural Network-Based Results:
#  The code generates plots comparing expected results to those produced by the 
#  neural networks for the sequences in the test set.

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

from functions import *
from input_file import *
import time
import tensorflow as tf

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create the scalers
x_scaler =  MinMaxScaler()
y_scaler  =  MinMaxScaler()

# Create loss functions
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()

TYPE_STRAINS_LIST = ["control_points", "random"]
mae_nn_random = []
mse_nn_random = []
mae_nn_control_points = []
mse_nn_control_points = []

for type_strains in TYPE_STRAINS_LIST:
    # Write the strains
    if (type_strains == "control_points"):
        writestrains(type_strains + STRAINS_PATH, MEAN_DEFORMATION, 
                      NUMBER_RANDOM_POINTS, NUMBER_INTERPOLATION_POINTS)
    
    elif (type_strains == "random"):
        write_random_strains(type_strains + STRAINS_PATH, MEAN_DEFORMATION, 
                      NUMBER_RANDOM_POINTS, NUMBER_INTERPOLATION_POINTS)
        
    # Write the stresses
    calculate_stresses_for_all(type_strains + STRAINS_PATH, 
                               type_strains + STRESS_FOLDER_PATH)
    
    # Create the dataset
    sequence_lenght = NUMBER_RANDOM_POINTS*NUMBER_INTERPOLATION_POINTS+1
    [x_train, y_train, x_validation, y_validation, x_test, y_test] =  \
        create_datasets(sequence_lenght, NUMBER_STRAIN_SEQUENCES, 
                        type_strains + STRAINS_PATH,  type_strains + 
                        STRESS_FOLDER_PATH, SPLIT)
        
    # Plot acc. Strain
    plot_acc_strain(type_strains + STRAINS_PATH + "/acc_strains/", x_train, 10)
    
    # Scale the dataset
    [x_train_scaled, y_train_scaled, x_validation_scaled, y_validation_scaled, 
      x_test_scaled, y_test_scaled] =  create_scaled_dataset(x_scaler, 
                                                             y_scaler,  
                                                             x_train, y_train, 
                                                             x_validation, 
                                                             y_validation, 
                                                             x_test, y_test)
    
    TYPES_NN = ["CNN", "RNN", "ENC_DEC", "CONV_REC_NN"]
    for type_nn in TYPES_NN:
        if(type_nn == "RNN"):
            NN = create_simple_rnn(10)
            pass
        
        elif (type_nn == "CNN"):
            NN = create_simple_cnn(20)
            
        elif (type_nn == "CONV_REC_NN"):
            NN = create_conv_rec_nn(20)
        
        elif (type_nn == "ENC_DEC"):
            NN = create_encoder_decoder(30)    
    
        # Compile the RNN
        compile_nn(NN, "Adam", "MSE")

        print("\n Training " + type_nn)
    
        # Train the NN 
        if(type_nn == "ENC_DEC"):
            model_history = train_enc_dec(NN, x_train_scaled, y_train_scaled, 
                                 x_validation_scaled,  y_validation_scaled, 
                                 x_test_scaled, y_test_scaled, 
                                 NUMBER_OF_EPOCHS) 
            # Predict y from x_test with the trained NN
            y_prediction_scaled = predict_enc_dec(NN, x_test_scaled)
            
        else:
            model_history = train_nn(NN, x_train_scaled, y_train_scaled, 
                                 x_validation_scaled,  y_validation_scaled, 
                                 x_test_scaled, y_test_scaled, 
                                 NUMBER_OF_EPOCHS)
        
            # Predict y from x_test with the trained NN
            y_prediction_scaled = NN.predict(x_test_scaled)
        
        y_prediction = y_scaler.inverse_transform(y_prediction_scaled.reshape([
                         y_prediction_scaled.shape[0] * 
                         y_prediction_scaled.shape[1],
                         y_prediction_scaled.shape[2]]))
        y_prediction = y_prediction.reshape(y_prediction_scaled.shape)
        
        
        if (type_strains == "control_points"):
            mae_nn_control_points.append(mae(y_test, y_prediction).numpy())
            mse_nn_control_points.append(mse(y_test, y_prediction).numpy())
        
        elif (type_strains == "random"):
            mae_nn_random.append(mae(y_test, y_prediction).numpy())
            mse_nn_random.append(mse(y_test, y_prediction).numpy())
            
        # Plot expected and calculated results
        plot_results(x_test, y_test/1e9, y_prediction/1e9, type_strains + 
                     "/Results/" + (type_nn + "/"), model_history)
        
        # Save the NN
        NN.save(type_strains + "/Results/" + type_nn + "/ML_model.h5")
        
        # Save the dataset
        serialise_dataset(x_train, y_train, x_validation, y_validation, x_test,  
                          y_test, y_prediction, (type_strains + "/Results/" +
                                                 type_nn + "/"))
    
with open('report.txt', 'a') as f:
    f.write("Results are the following: \n \n")
    f.close()
    
for cont in range(len(TYPES_NN)):
    # if (type_strains == "control_points"):
    with open('report.txt', 'a') as f:
        f.write("MAE(expected vs. predicted) using control points and " + 
                TYPES_NN[cont] + " is " + str(mae_nn_control_points[cont]) + 
                "\n")
        f.write("MSE(expected vs. predicted) using control points and " + 
                TYPES_NN[cont] + " is " + str(mse_nn_control_points[cont]) + 
                "\n \n")
        f.write("MAE(expected vs. predicted) using random strains and " + 
                TYPES_NN[cont] + " is " + str(mae_nn_random[cont]) + "\n")
        f.write("MSE(expected vs. predicted) using random strains and " + 
                TYPES_NN[cont] + " is " + str(mse_nn_random[cont]) + "\n \n")
        f.close()

print("\n End")
