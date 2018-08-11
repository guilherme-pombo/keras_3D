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

After 10 epochs:
```bash
loss: 0.2803 - acc: 0.9168 - val_loss: 0.5536 - val_acc: 0.8290
```

Then to train a multi view model use

```bash
python3 train_multiview_resnet.py --model_type concat --batch_size 32 --epochs 10
```