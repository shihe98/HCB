# HCB
### ABOUT
This is the training source code for two threat models of HCB.
There are two scenarios: data outsourcing and model outsourcing.

You can access training and test datasets at [this link](https://drive.google.com/drive/u/0/folders/13R4eBqMfWAGeM0bN3m38bOco2l2j3ljA). The datasets include non-rainy images and rainy images, representing non-effective samples and effective samples, respectively. To run the code successfully, please ensure that you place these files in the "data" folder within the root directory.

This is for releasing the source code of our work "Watch Out! Simple Horizontal Class Backdoor Can Trivially Evade Defense". If you find it is useful and used for publication. Please kindly cite our work as:
```python
@inproceedings{ma2024hcb,
  title={Watch Out! Simple Horizontal Class Backdoor Can Trivially Evade Defense},
  author={Ma, Hua and Wang, Shang and Gao, Yansong and Zhang, Zhi and Qiu, Huming and Xue, Minhui and Abuadbba, Alsharif and Fu, Anmin and Nepal, Surya and Abbott, Derek},
  booktitle={2024 ACM Conference on Computer and Communications Security (CCS)},
  year={2024},
  organization={ACM}
}
```


### DEPENDENCIES
Our code is implemented and tested on PyTorch. Following packages are used by our code.
- `troch==1.7.1`
- `torchvision==0.8.2`
- `numpy==1.19.5`

### RUN
1. The training and test phases on the data outsourcing scenario.
```python
python GTSRB/Data/HCB_Data_Outsourcing.py
```
2. The training and test phases on the model outsourcing scenario.
```python
python GTSRB/Model/HCB_Model_Outsourcing.py
```

### Other Tasks
To be continue...
