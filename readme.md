# Asteriod

## Intro

Asteriod是pytorch下的一个语音分离库。

- hardware_cola是虚拟环境，放在你的envs下
- signal create用于生成模拟信号（目标1到目标42），需用matlab2012，想试试的话可以自取
- Sanya是主项目，数据我已经处理完毕，放在train和val下，metadata是标签文件, 数据集格式：
  - `s1`和`s2`为两种纯净信号（我目前试的目标1和目标2）
  - `noise`为噪声信号
  - `mix_clean`为`s1`,`s2`混合
  - `mix_both`为s1，`s2`，`noise`混合

工程代码在code中
数据集生成过程为：
data82.py用于把数据按8：2分到train和val的s1，s2中
generate_noise.py生成噪声
rename.py把数据名替换成英文
mixture_clean.py和mixture_both.py用于生成mix_clean和mix_both下的数据
generate_label.py用于生成metadata中的标签

自己运行前先修改generate_label.py中的s1_path和s2_path，再把生成的csv分别放入metadata的train和val
最后train.py训练

由于matlab生成的信号只有前0.09s代表调频和cw特征，所以cut.py用于分割信号

## Sanya Todo

1. `train.py`: mode `enh_single`
2. dataset folder:

## Sanya UI TODO

1. [x] load model
2. [x] si-snr
3. [ ] ui
