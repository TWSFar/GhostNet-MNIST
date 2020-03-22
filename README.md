# GhostNet-MNIST
***基于Python的手写数字识别系统***
## Install
```shell
git clone git@github.com:TWSFar/GhostNet-MNIST.git
pip install -r requirements.py
```

## Train
```python
1. Configure training parameters in 'config.py'
2. python train.py  # Training
```

## Test
```python
python test.py --image ${IMAGE PATH OR DIRECTORY} --checkpoint ${CHECKPOINT_FILE}
```
Optional arguments:
- `--show`: Show image and Result

Examples:

1. Test one image 
```python
python test.py --image 'demo/1.jpg' --checkpoint 'work_dir/model_best.pth' --show
```

2. Test multiple images
```python
python test.py --image 'demo' --checkpoint 'work_dir/model_best.pth' --show
``` 