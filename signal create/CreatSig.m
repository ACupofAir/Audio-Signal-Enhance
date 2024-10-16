clear all
close all
clc
fs=125e3;


%%%%调频1参数%%%%
fl=[14e3,16e3,16e3,18e3,18e3,20e3];
fh=[16e3,14e3,18e3,16e3,20e3,18e3];

%%%%CW1%%%%%
f_cw1=[14e3:1e3:20e3];

%%%%%CW2%%%%%%
f_cw2=[14.5e3:1e3:20.5e3];

%%%%%%
[y1,~] = hfm(fl(1),fh(1),fs,0.03);
[y2,~] = hfm(fl(2),fh(2),fs,0.01);
HFM1_1=[y1,y2];
[y1,~] = hfm(fl(2),fh(2),fs,0.03);
[y2,~] = hfm(fl(1),fh(1),fs,0.01);
HFM1_2=[y1,y2];
[y1,~] = hfm(fl(3),fh(3),fs,0.04);
HFM1_3=y1;
[y1,~] = hfm(fl(4),fh(4),fs,0.04);
HFM1_4=y1;
[y1,~] = hfm(fl(5),fh(5),fs,0.03);
[y2,~] = hfm(fl(6),fh(6),fs,0.01);
HFM1_5=[y1,y2];
[y1,~] = hfm(fl(6),fh(6),fs,0.03);
[y2,~] = hfm(fl(5),fh(5),fs,0.01);
HFM1_6=[y1,y2];

HFM_A=[HFM1_1;HFM1_2;HFM1_3;HFM1_4;HFM1_5;HFM1_6];
tCount=0;
t=0:1/fs:0.02-1/fs;
for ii=1:length(f_cw2)
    for jj=ii+1:length(f_cw2)
        tCount=tCount+1;
        CW2(tCount,:)=sin(2*pi*f_cw2(ii).*t)./2+sin(2*pi*f_cw2(jj).*t)./2;
    end
end

for ii=1:length(f_cw1)
    CW1(ii,:)=sin(2*pi*f_cw1(ii).*t);
end
tao1=0.01:0.01:0.1;
tao2=0.01:0.01:0.15;

tarNum=0;  Sig_N=[];
for iHFM=1:length(HFM_A(:,1))
    HFM_N=HFM_A(iHFM,:);
    for iCW1=1:length(CW1(:,1))
        tarNum=tarNum+1;
        CW1_N=CW1(iCW1,:);
        for ii=1:length(tao1)
            for iCW2=1:length(CW2(:,1))
                CW2_N=CW2(iCW2,:);
                for jj=1:length(tao2)
                    Sig_N=[HFM_N,zeros(1,0.03*fs),CW1_N,...
                        zeros(1,tao1(ii)*fs),HFM_N,zeros(1,0.03*fs+0.1*fs-tao1(ii)*fs),...
                        CW2_N,zeros(1,tao2(jj)*fs),HFM_N,zeros(1,0.15*fs-tao2(jj)*fs)];
                    figure(1);
                    plot((1:length(Sig_N))./fs,Sig_N);
                    xlabel('时间/s');
                    ylabel('幅度/v');
                    title(['目标',num2str(tarNum),'第',num2str(ii),'个周期，第',num2str(iCW2),'个潜标组，深度',num2str(tao2(jj)*1000*10),'m'])
                    saveas(gcf,['目标',num2str(tarNum),'第',num2str(ii),'个周期，第',num2str(iCW2),'个潜标组，深度',num2str(tao2(jj)*1000*10),'m.tif']);
                    wavwrite(Sig_N,fs,['目标',num2str(tarNum),'第',num2str(ii),'个周期，第',num2str(iCW2),'个潜标组，深度',num2str(tao2(jj)*1000*10),'m.wav']);
                end
            end
        end
    end
end


