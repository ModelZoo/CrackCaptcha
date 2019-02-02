# CrackCaptcha

Crack Geetest and Dun163 Sliding Captcha by Deep Learning.

## Usage

You need to install [Git LFS](https://git-lfs.github.com/) before using this repository.

Firstly, you can clone this repo by following commands:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ModelZoo/CrackCaptcha.git
cd CrackCaptcha
git lfs pull
```

After above commands, you can get all of the codes and dataset in your computer.

Then make sure you've installed the proper packages for this repository:

```
pip3 install -r requirements.txt
```

Then you can train with downloaded dataset using this command:

```
python3 train.py
```

After this command, the training process will be started.

Checkpoints will be saved at `checkpoints` folder, TensorBoard Events will be saved at `events` folder.

But this model get no good...

## Loss

Loss transition:

![](https://ws3.sinaimg.cn/large/006tNbRwgy1fyaxat4h67j30sd0ixmxx.jpg)

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fyaxb21on3j30st0itwfd.jpg)

