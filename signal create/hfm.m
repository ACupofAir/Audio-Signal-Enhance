function [y,iflaw] = hfm(F1,F2,Fs,T,type)
% % ��������������˫����Ƶ�ź�
% F1    �� ˫����Ƶ�źŵ���ʼƵ��
% F2    �� ˫����Ƶ����ֹƵ��
% Fs    �� �źŵ�����Ƶ�ʣ�
% T     �� �źŵĳ���ʱ��
% y     �� ˫����Ƶ�źţ�
% type  �� �ź����� 1��ʾ��ʵ�źţ���������Ǹ��źţ�Ĭ�ϲ���ʵ�ź�
% iflaw �� �źŵ�˲ʱƵ��
% Example��
% [y,iflaw]=hfm(1750,500,6000,2,1);
% specg(y,6000);  pause; plot(iflaw);

if nargin<4
    error('˫����Ƶ�ź�������Ҫ4��������');
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