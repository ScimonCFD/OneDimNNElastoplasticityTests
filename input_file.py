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
# Material properties and other required constants

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

# Material properties
E = 200e9 # Young's modulus
K = 300e6 # Plastic slope
SIGMA_Y = 1e9 # Yield stress

#Stress state of the material
eps_current = 0 #Epsilon at time n
eps_p_current = 0 #Epsilon plastic at time n
alpha_current = 0 #Alpha_n
sigma_current = 0 #Sigma_n

sigma_results = [sigma_current]   # Stresses
eps_results = [eps_current]       # Strains

# Constants
NUMBER_STRAIN_SEQUENCES = 10000
MEAN_DEFORMATION = 0.025
PARENT_FOLDER_PATH ='sequences'
STRAINS_PATH = '/strain_sequences' #PARENT_FOLDER_PATH +'/strain_sequences'
STRESS_FOLDER_PATH = '/strain_stress_sequences'#PARENT_FOLDER_PATH + '/strain_stress_sequences'
NUMBER_RANDOM_POINTS = 10#15#10
NUMBER_INTERPOLATION_POINTS = 10#5
SPLIT = [0.7, 0.2, 0.1]
SEED = 0
NUMBER_OF_EPOCHS = 1000
# TYPE_STRAINS = "control_points"#"random" # "control_points"
# TYPE_STRAINS = "random"