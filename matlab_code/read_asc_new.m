
function read_asc(filename)
addpath('../../0ng/export');
addpath('../../0ng/');
one_or_all=2;  %%1 for only 1 pixel 2 for all
if(nargin==0)
    filename='00ng_im2_NADH_PC102_60s_40x_oil';
end

%%Reading fitted data2
fname_a1=strcat(filename,'_a1.asc');
fname_a2=strcat(filename,'_a2.asc');
fname_t1=strcat(filename,'_t1.asc');
fname_t2=strcat(filename,'_t2.asc');

A1=double(dlmread(fname_a1));
A2=double(dlmread(fname_a2));
T1=double(dlmread(fname_t1));
T2=double(dlmread(fname_t2));
Nsize=256;
n=Nsize;
tp=Nsize;
O=double(ones(n,n));


[BWmask im]=read_intensity_image(filename,Nsize);

BWmask_vec=BWmask(:);
[a b]=find(BWmask);
pixels_roi_list=[a b];

%%generate the exponential fit
F=zeros(n,n,tp);
for t=1:1:tp
    temp=t*10;
     F(:,:,t)=(A1.*exp(-temp*O./T1)+A2.*exp(-temp*O./T2));
     
end

% % %%reading instrument response
A=xlsread('madh_urea_40x_60s_IRF.xlsx');
irf_org=A(:,2);
maxI=max(irf_org);
irf=irf_org./norm(irf_org);%%normal=
ind=find(irf>0);
irf_cropped=irf(ind);

 irf=irf;%%set to irf_org
%irf=irf_org;


bin=2;
margin=bin+1;
N1=Nsize-2*margin+1;
M1=N1;


I=zeros(n-2*margin-1,n-2*margin-1,tp);
%%convolving irf and firf(signal)
for i=margin:n-margin
    for j=margin:n-margin
%         z=myconv(reshape(F(i,j,:),[1 tp]),irf_cropped');
        z=myconv(reshape(F(i,j,:),[1 tp]),irf');
       
        
        I(i-margin+1,j-margin+1,:)=z;
    end      
end    

%%removing curves which belong to the background
x=186;
y=93;
Index=[];%%list of pixels on which to run the model, with background eliminated
newim=zeros(N1,M1);
Label=[];
index=0;
Max_v=max(I,[],3);
Max_v=Max_v(:);
display('Range of values of signal(max)');
[min(Max_v) max(Max_v)]

%%creating labels
if(one_or_all==2)
    for i=1:tp-2*margin-1
        for j=1:tp-2*margin-1
            mv=max(I(i,j,:));
            %[i j mv]
            if(i==186) && (j==93)
              index=size(Index,1)+1;
            end
         if(mv>.004)%%.011
    
                Index=[Index; [i j]];
                if(mv>.005) && (im(i,j)>50)%%15
                    Label=[Label; 1];
                    newim(i,j)=255;
                
                else
                    Label=[Label; 0];
                    newim(i,j)=100;
                 %pause
             end
           % 
           %pause
         end
        end
    end
else
    Index=[186 93];
end    
 imshow(uint8(newim))
display('Number of pixel classifying');
size(Index,1)
display('Number of label of each class')
[numel(find(Label==1)) numel(find(Label==0))]

ind_list=sub2ind([Nsize Nsize],Index(:,1),Index(:,2));
F=reshape(F,Nsize*Nsize, Nsize);

I=reshape(I,N1*M1, Nsize);%%for bin=2
F=F(ind_list,:);
max(ind_list)
size(I)
size(ind_list)
I=I(ind_list,:);

%Label=BWmask_vec(ind_list);
%Label(find(Label==0))=-1;%%changing labels to 1 and -1


im_vec=im(:);
% 

% 
 [B V]=construct_laguerre_bases(F,I,n,tp,irf');
%  [size(V) size(I)]
%  pause
 %fit_signal(B,V,I,[186 93]);
 %fit_signal(B,V,I);
 
 %I(index,:)for 1
 regress_classify(B,V,I,[Index Label]);
 dlmwrite('../../0ng/export_modified/train/im2/label.txt',[Index Label],',');
% 
% 
% % %%Reading the raw data
% % 
% % Data=(bfopen(strcat('../0ng/',strcat(filename,'.sdt'))));
% % Data=Data{:,1}
% % D=[];
% % D=uint16(D);
% % for i=1:Nsize
% % D=cat(3,D,(Data{i,1}));
% % end
% %plot10(D,tp);
% 



%F raw, not convolved
%S/I convolved data

function [imbw im]=read_intensity_image(filename,Nsize)
bin=2;
margin=bin+1;
im=imread(strcat(filename,'_intensity_image.bmp'));
im=rgb2gray(im);
im=flipud(im);%%added to match with the corordinate in the software
im=imresize(im,[Nsize Nsize]); %% check with sagar
imbw=zeros(size(im));
imbw(im>=10)=1;%%35 earlier
[N M]=size(imbw);
imbw=imbw(margin:N-margin,margin:M-margin);
imwrite(imbw,'../../0ng/export_modified/train/im2/00ng_im2_NADH_PC102_60s_40x_oil_truth.bmp')



function plot10(S,tp,irf,F)
% %%if random
% x=randperm(Nsize);
% y=randperm(Nsize);
% x=x(1:10);
% y=y(1:10);

%%chose two points which are not in the background
% x=[47 186];
% y=[219 93];
x=186;
y=93;
for i=1:numel(x)
    close all;
   %% [x(i) y(i)];
    y1=reshape(S(x(i),y(i),:),[1 tp]);
    
    semilogy(y1,'r-')
    %figure,
    if(nargin>2)
        hold on;
%          semilogy((irf),'g-');
%          hold on;
        %figure,
        %reshape(F(x(i),y(i),:),[1 tp])
        semilogy(reshape(F(x(i),y(i),:),[1 tp]),'b-');
        legend('fitted function convolved with irf','fitted exponential');
       % hold off;
    end
     
    
end