# Dynamic Graph CNN for Learning on Point Clouds (PyTorch)
This repo is implemented upon and has the same dependencies as the official [DGCNN repo](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) .
## Point Cloud Classification with BoW pooling
* Run the training script with bow pooling:

``` 1024 points
python main.py --exp_name=dgcnn_1024 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True
```

``` 1024 points
python main.py --exp_name=dgcnn_1024 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --mixing_update=True --T=50
```

* Run the evaluation script with our pretrained models. Download the our [pretrained models](https://drive.google.com/file/d/12TbAVavYBRHt2w7I2--0U3jatQUmr5AJ/view?usp=sharing), change the --mode_path to the PATH you download model.t7:

``` 1024 points
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=PATH/model.t7
```


