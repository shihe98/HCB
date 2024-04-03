# HCB
### ABOUT
This is the training source code for two threat models of HCB.
There are two scenarios: data outsourcing and model outsourcing.

You can access training and test datasets at . The datasets include non-rainy images and rainy images, representing non-effective samples and effective samples, respectively. To run the code successfully, please ensure that you place these files in the "data" folder within the root directory.

This is for releasing the source code of our work "DeepTheft: Stealing DNN Model Architectures through Power Side Channel".

### DEPENDENCIES
Our code is implemented and tested on PyTorch. Following packages are used by our code.


### RUN
1. The training and test phases on the data outsourcing scenario.
python GTSRB/DataOutsourcing/HCB_Data_Outsourcing.py
2. The training and test phases on the model outsourcing scenario.
python GTSRB/ModelOutsourcing/HCB_Model_Outsourcing.py
