function [y,iflaw] = hfm(F1,F2,Fs,T,type)
% % 产生给定参数的双曲调频信号
% F1    ： 双曲调频信号的起始频率
% F2    ： 双曲调频的终止频率
% Fs    ： 信号的中心频率；
% T     ： 信号的持续时间
% y     ： 双曲调频信号；
% type  ： 信号类型 1表示是实信号，否则产生是复信号，默认产生实信号
% iflaw ： 信号的瞬时频率
% Example：
% [y,iflaw]=hfm(1750,500,6000,2,1);
% specg(y,6000);  pause; plot(iflaw);

if nargin<4
    error('双曲调频信号至少需要4个参数。');
elseif nargin==4
    type=1;
end
F0 = (F1+F2)/2;
B = F2-F1;
t0 = T*F0/B;
K = (F0-B/2)*(t0+T/2);
N = floor(Fs*T);
t = [0:N-1]/Fs;
if isequal(type,1)
    y = cos(2*pi*K*log(1-(t-T/2)/t0));
else
    y = exp(-j*2*pi*K*log(1-(t-T/2)/t0));
end
iflaw = K./(t0-(t-T/2));