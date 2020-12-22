## BoW Pooling: A Plug-and-Play Unit for Feature Aggregation of Point Clouds
Created by Xiang Zhang, Xiao Sun, Zhouhui Lian from Peking University.

![pipeline](figures/pipeline.png)
### Introduction

This work will appear in AAAI 2021. We propose the BoW pooling, a plug-and-play unit that substitutes for the symmetric functions in existing methods for the feature aggregation of point clouds. A novel dictionary update strategy is explored and discussed. The truncated Linear Unit is introduced to suppress the expression of unimportant local descriptors. You can also check out [paper]() for a deeper introduction.

Point cloud provides a compact and flexible representation for 3D shapes and recently attracts more and more attentions due to the increasing demands in practical applications. The major challenge of handling such irregular data is how to achieve the permutation invariance of points in the input.


In this repository, we release the code and data for BoW pooling for point cloud classification, shape retrieval and segmentation tasks.

### Citation

if you find our work useful in your research, please consider citing:

```

```

### Installation

The code has been tested with Python 3.6, PyTorch 1.0.0, Tensorflow 1.12.0 and CUDA 9.0 on Ubuntu 16.04.

### Usage

##### Data Preparation
()
Firstly, you should download the [ModelNet40 dataset](https://drive.google.com). [SHREC15 dataset](). [ShapeNet part dataset](). [S3DIS dataset]().
```

```

##### Train Model

To train and evaluate BoW pooling for classification:

```bash
python train.py
```

##### Test Model

The pretrained backbone model with BoW pooling weights are stored in [pretrained model](https://drive.google.com). You can download it.

To evaluate the BoW pooling for classification:

```bash
python test.py
```

### Licence

Our code is released under MIT License (see LICENSE file for details).
