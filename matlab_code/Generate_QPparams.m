function x=Generate_QPparams(H,f,A,b,G,h,m,Label);
%%% sum(sum((V*X-B).*(V*X - B))) == trace(X'*V'*V*X-2*B'*V*X+B'*B)


%%M is the number of columns of the variable matrix. 
if(nargin<1)
%%variable should be n*m
    n=10;
    m=64;
    
    %%reading presaved mat file
    
%      load H.mat
%     H=HH;
%     G=D;
%     A=[];
%     b=[];
    alpha=0;
    V=rand(n-2,n);
    [U S]=eig(V'*V);
    
    H=U*abs(S)*U';
    H=(H+H')/2;
    f=rand(m,n);
    A=zeros(1,n);
    b=zeros(1,1);
    G=rand(4,n);
    G=ones(1,4)*G;
    h=zeros(1,1);
    Label=sign(rand(m,1)-.5);
    Label(find(Label==-1))=0;
    L=(Label*Label');
    LG=diag(sum(L,1))-L;
    Label=reshape(Label,[8 8]);
    %dlmwrite('label.txt',Label,',');
    %H=H+alpha*LG; %objective is tr(alpha'*H*alpha) + trace(alpha*L*alpha')
    %where alpha is what we are solving for
end
if(nargin>4)
    G=ones(1,size(G,1))*G;
end
if(m>1)
    n=size(H,1);
    
    %Hkron = kron(eye(m),H);
    f1=f';
    f_save=f;
    f=f1(:);
    A=zeros(1,n*m);
    b=zeros(1,1);
    
    G_save=G;
    G=(kron(eye(m),G));
    
    
   
end
% [size(H) size(f)]
% pause
h=zeros(size(G,1),1);
    
n=size(H,1);

if(numel(A)==0)
    A=zeros(1,n);
end
if(numel(b)==0)
    b=zeros(1,1);
end

A(end)=.0000000001;
HH=H;
if(m==1)
    D=G;
else
    D=G';
    
end
x=rand(n,m);

save('../../0ng/export_modified/train/im2/H.mat','-v7.3','HH', 'f','D','m');
if(m>1)
    f=f_save;
    G=G_save;
end
%rank([H; A; G])

%%calling cvxopt
%  !python cvxopttry.py
% load solution.mat
% x1=x/2;
% if(m>1)
%     x1=reshape(x1,[n m]);
% end
% [size(H) size(f) size(x1) size(G)]
% 
% if(size(f,1)==1)
%     obj1= trace(.5*x1'*H*x1 +f*x1);
% else
%     obj1= trace(.5*x1'*H*x1 +f*x1);
% end
    
%cons1=G*x1-h

% %calling cvxpy
% !python cvxpytry.py
% load solution_cvxpy.mat
% x2=x'-.006;
%if(m>1)
   % x2=reshape(x2,[n m]);
%end

% if(size(f,1)==1)
%     obj2= .5*x2'*H*x2 +f*x2
% else
%     obj2= .5*x2'*H*x2 +f'*x2
% end
%x_qp= quadprog(2*H,f',G,h);

% !python ../qpth-master/test1.py
%  load solution_qpth.mat
% x3=x'/2;
% if(m>1)
%     x3=reshape(x3,[n m]);
% end
% 
% 
% if(size(f,1)==1)
%     obj3= trace(.5*x3'*H*x3 +f*x3)
% else
%     obj3= trace(.5*x3'*H*x3 +f*x3)
% end
% [obj3]
% [x3]


% 
% %load('resultx.mat');
% %x3=x';
% %obj3= .5*x3'*H*x3 +f*x3
% %cons2=G*x3-h
% [x1 x2]
% obj1
% obj2
 %x=x3;
