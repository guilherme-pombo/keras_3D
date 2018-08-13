# keras_3D

First download the data from:

```bash
wget https://www.cs.virginia.edu/~connelly/dump/modelnet40_data.tar.gz
tar -zxvf modelnet40_data.tar.gz
```

Then to train a single view model use

```bash
python3 train_singleview_resnet.py --batch_size 32 --epochs 10
```

Then to train a multi view model use

```bash
python3 train_multiview_resnet.py --model_type mvcnn --batch_size 32 --epochs 10
```