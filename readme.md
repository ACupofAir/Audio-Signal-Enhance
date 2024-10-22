# Asteriod

## Intro

Asteriod是pytorch下的一个语音分离库。

- hardware_cola是虚拟环境，放在你的envs下
- signal create用于生成模拟信号（目标1到目标42），需用matlab2012，想试试的话可以自取
- Sanya是主项目，数据放在train和val下，metadata是标签文件, 数据集格式：
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

### 数据集制作

1. 制作数据集`data_82.py`: 首先需要把数据转为等长的wav信号，分入train和val的s1和s2中；假设总共1000条，则train/s1和train/s2中各400条，val/s1和val/s2各100条；
2. 噪音生成`generate_noise.py`: 用于生成noise下的高斯噪声，采样率和长度与信号相同；数量和s1/s2中数量相同，即train/noise中400条，val/noise中100条；
3. 纯净混合信号生成`mixture_clean.py`: 用于生成mix_clean下的数据（可能要根据实际的数据情况稍作修改）
4. 所有混合信号生成`mixture_both.py`: 用于和mix_both下的数据 （可能要根据实际的数据情况稍作修改）
5. 生成数据标签`generate_label.py`: 用于在metadata下生成标签，把生成的csv分别放入metadata的train和val （可能要根据实际的数据情况稍作修改）

### 训练

1. `train.py`训练:
    - `n_src`改为`2`
    - `sample_rate`改为采样率
    - `task`改为`enh_both`，这样对`s1`和`s2`同时增强
    - 分段数`segment`修改为：segment*sample_rate尽可能接近（可等于但不能超过）csv的最后一列`length`，即每段wav的采样点数；例如采样率125000，每段wav采样点数length为250000，则`segment`最大可取2

### Sanya TODO List

1. [x] load model
2. [x] si-snr
3. [x] ui
4. [ ] train workflow
