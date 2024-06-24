## Assessing Neural Networks in Predicting Stress-Strain Behavior with Isotropic Hardening

### Description

This Python code is designed for engineers and researchers in materials science to explore the potential of neural networks in predicting stress-strain behavior in materials exhibiting 1D isotropic hardening. The code consists of three main components:

1. **Generation of 1D Strain Sequences:**
   - The code generates a set of 1D strain sequences, which represent the deformation of a material under load. These sequences can be customized for specific test scenarios or data requirements.

2. **Calculation of Corresponding Stresses with Isotropic Hardening:**
   - Utilizing an isotropic hardening model, the code calculates the corresponding stresses for the generated strain sequences. Isotropic hardening is a common material behavior where a material's yield strength increases with plastic deformation.

3. **Evaluation of Neural Networks:**
   - The code implements various types of neural networks, such as feedforward neural networks, recurrent neural networks (RNNs), or convolutional neural networks (CNNs). These networks are trained and tested using the generated stress-strain data.
   - The goal is to assess the performance of neural networks in approximating the stress-strain behavior, and ultimately, to evaluate if they can serve as a replacement for the theoretical material model. Metrics such as mean squared error and mean absolute error are computed to gauge the accuracy of the neural network predictions.
   - Researchers can fine-tune the neural network architectures, hyperparameters, and training datasets to optimize their performance and explore the potential for more efficient material modeling.

This code provides a valuable tool for investigating the feasibility of using neural networks to predict stress-strain behavior in materials exhibiting isotropic hardening, potentially offering a faster and more versatile alternative to traditional material models.

### How do I get set up? ###

The following libraries are required:

* Python 3.8.12

* NumPy 1.18.5

* TensorFlow 2.4.0

* pip 21.3.1

* Matplotlib 3.3.1

* scikit-learn 1.0.1

* pandas 1.1.1

* tqdm 4.50.2

These libraries can be installed from the supplied environment.yml file using the conda software (https://conda.io). Once conda is installed, the Python environment is installed with:

    conda env create -f environment.yml

The conda environment can be activated with:

    conda activate pythonPal-no-gpu

Then, clone this folder:

    git clone git@github.com:ScimonCFD/1DNNCodeForThesis.git

Finally, the code can be run with:

    python main.py

## Some results ##

<figure>
  <img src="https://github.com/ScimonCFD/1DNNCodeForThesis/blob/master/img/CNN_1D_1.png" alt="">
  <figcaption>Expected (green) vs predicted (black) behaviour. The prediction was made using a convolutional neural network. </figcaption>
</figure>

<figure>
  <img src="https://github.com/ScimonCFD/1DNNCodeForThesis/blob/master/img/RNN_1D_1.png" alt="">
  <figcaption>Expected (green) vs predicted (black) behaviour. The prediction was made using a neural network. </figcaption>
</figure>

<figure>
  <img src="https://github.com/ScimonCFD/1DNNCodeForThesis/blob/master/img/ENC_DEC_1D_1.png" alt="">
  <figcaption>Expected (green) vs predicted (black) behaviour. The prediction was made using a simple encoder-decoder.</figcaption>
</figure>

<figure>
  <img src="https://github.com/ScimonCFD/1DNNCodeForThesis/blob/master/img/CONV_REC_1D_1.png" alt="">
  <figcaption>Expected (green) vs predicted (black) behaviour. The prediction was made using a mixed recurrent + convolutional neural network.</figcaption>
</figure>




### Who do I talk to? ###

    Simon Rodriguez
    simon.rodriguezluzardo@ucdconnect.ie
    https://www.linkedin.com/in/simonrodriguezl/
    
    Philip Cardiff
    philip.cardiff@ucd.ie
    https://www.linkedin.com/in/philipcardiff/
