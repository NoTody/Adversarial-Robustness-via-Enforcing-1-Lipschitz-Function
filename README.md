# Adversarial-Robustness-via-Enforcing-1-Lipschitz-Function
Project for NYU Foundations of Machine Learning CSCI-GA 2566 against Adversarial examples generated by AutoAttack from CIFAR-10. Our method achieve 59.50% evaluated on 1000 samples of standard mode AutoAttack, which outperforms SOTA method trained without synthetic or extra data on RobustBenchmark (88.04%).

## To install
```
pip install -r requirement.txt
```
For important changes to make to run the code, please check Dependencies.txt file

## To train our proposed method with Spectral Normalization and Stochastic Weights Averaging
The below command run with 2 gpus, 256 batch size, 8 workers, 200 epochs, sgd optimizer, Jenson Shannon loss, WResNet34-10 as backbone, learning rate 0.1, multi-step scheduler while using Spectral Normalization and Stochastic Weights Averaging
```
python cr_train.py --batch_size 256 --num_workers 8 --max_epochs 200 --gpus 2 --optimizer "sgd" --loss_func "JS" --backbone_model "WResNet" --lr 0.1 --scheduler "multistep" --use_sn --use_swa --runpath "./runs" --model_dir "JS_sn_swa"
```
For more argument settings, please check argument parser help and our detailed argument setup at cr_train.py and cr_pl.py

## To test result with loaded model by AutoAttack
The below command run with 1 gpus, 8 workers, WResNet34-10 as backbone, load path to be "./runs/JS_swa_sn/epoch=89-step=8820.ckpt" and "./runs/MIN_swa_sn/swa.ckpt" with ensemble enabled (Note: if multiple models are loaded, their load paths are spearated by ", " while passing into --load_path argument and --ensemble need to be set), evaluted on full testset (Note: if num_examples set to -1, full testset is evaluated) and no models are not using spectral norm (Note: if some models to be loaded don't use spectral need to passing in indeces of the models that don't use spectral norm to --no_sn. For example, --no_sn "0, 1" indicate first two loaded models from --load_path are not using spectral norm, "" indicates all models are using spectral norm)
```
python cr_test.py --num_workers 8 --gpus 1 --backbone_model "WResNet" --load_path "./runs/JS_swa_sn/epoch=89-step=8820.ckpt, ./runs/MIN_swa_sn/swa.ckpt" --ensemble --use_sn --num_examples -1 --no_sn ""
``` 

This will create tensorboard file where you can monitor the training schedule. In the tensorboard, you can find thhe graph for adversarial attack score from AutoAttack.

## To use tensorboard
```
tensorboard --logdir=./runs
```
## Correspondence
Haoxu Huang (hh2740@nyu.edu), Jiraphon Yenphraphai (jy3694@nyu.edu), Haozhen Bo (hb2432@nyu.edu)
