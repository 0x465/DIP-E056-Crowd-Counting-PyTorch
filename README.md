# DIP-E056-Crowd-Counting-PyTorch

This repo is the implementation of CSRNet Crowd Counting using PyTorch.

### Prerequisites
Anaconda environment is strongly recommended.   

The following libraries was used:
- Python 3 
- PyTorch 1.6.0
- OpenCV 3.4.2
- CUDA 10.2

### Dataset
Refer to following pages:
[ShanghaiTech Dataset](https://www.kaggle.com/tthien/shanghaitech-with-people-density-map)  
**NOTE**: Part_A includes more dense data and Part_B includes scattered data.

### Preprocessing Data  
Download and unzip ShanghaiTech folder into root directory.   
Root directory should have the following hierachy:    
**NOTE**: '*' files will only appear after generation.   
```
ROOT
  |-- shanghaitech_with_people_density_map
        |-- .....
  |-- json
        |-- part_A_train.json *
        |-- part_A_test.json *
        |-- part_B_train.json *
        |-- part_B_test.json *
  |-- models
        |-- checkpoint.pth.tar *
        |-- model_best.pth.tar *
  |-- CC.py
  |-- config.py
  |-- create_ground_truth.py
  |-- create_json.py
  |-- dataset.py
  |-- image.py
  |-- predict.py
  |-- train.py
  |-- utils.py
```
### Getting Started
Please ensure all required libraries is installed.    
Update paths in the following files:
```
- create_json.py
- create_ground_truth.py
- predict.py
- train.py
- utils.py
```

### Generate GroundTruth
Generate groundtruth and density map using ```create_ground_truth.py```.

### Generate JSON
JSON files can be created using ```create_json.py``` and will be saved to ROOT/json.

### Training & Testing
Run ```train.py``` to train the model. Model checkpoints will be saved to ROOT/models.

### Prediction
Prediction can be done on ShanghaiTech images or a new image using ```predict.py``` (update paths in file accordingly). An density map ```prediction.png``` will be generated and saved to ROOT/.


### References
This project was developed using the repo of [surajdakua](https://github.com/surajdakua/Crowd-Counting-Using-Pytorch) as a base.  

Please cite the Shanghai datasets and other works if you use them.
```
@inproceedings{zhang2016single,
  title={Single-image crowd counting via multi-column convolutional neural network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={589--597},
  year={2016}
}
```
