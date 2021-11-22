## CNN Image Quality Classifier

### Overview

The second approach was better than the first, at least I didn't feel like I'm cheating, yet I was not 
satisfied. Keras is a High-level Deep Learning library, so I decided to recreate the project in Torch.
Torch as a Lower-level library, allows you more control over the neural network and the functionality flow
in depth. The current architecture generalizes in ~5 epochs.

### Usage

If the model has already been saved, skip step number **2**.

1. Install the necessary requirements

`pip install -r requirements.txt`

2. Run the *training.py* script to create data files, train and save the model 

`python training.py`

3. Pass an image to the *predict.py* script, receive an output

`python predict.py --image 'PATH/TO/IMAGE'`

### Project architecture

This project contains multiple scripts:

1. **create_data_jsons.py** - Generate train and test json files with evenly distributed classes


2. **preprocess_data.py** - As the name suggests, the script preprocesses the training and testing images 
   (conversions into tensors, reshaping, resizing, etc.), but also *overrides* the **data.Dataset** util by torch 
   to load the json data in the specific format


3. **neuralnet.py** - Where we define the model, the hyperparameters and the classes


4. **training.py** - We bring things together. Here, we create, load and preprocess the data and feed it to
   the network. Each epoch we test the network to evaluate the progress and save the current model parameters.


5. **predict.py** - Loads the model, preprocesses the image passed as argument and returns a prediction. 


### Model performance

There are approx. ~69mil. trainable parameters and the neural network architecture of the model is very 
similar to the Keras model. In the first epoch, the model can't differentiate average and bad quality,
 which results in 66% accuracy on the test data (*2/3 classes*). In the second epoch, the loss drops by
half, and the model already results in 100% accuracy on the test data. However, using AdamW optimizer
with **5e-4** learning rate, the model achieves 100% accuracy and decent loss within the first epoch.


### Final touch

The code can be further optimized to generate the data faster, setup GPU to train faster, play with the
architecture to obtain smaller loss, introduce learning rate scheduler, etc. Having in mind
that this is a simple task, I assume that the current model fulfills the client's needs. 
