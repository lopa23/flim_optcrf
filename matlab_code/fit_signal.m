function fit_signal(b,V,I,pixel)
doplot=0;
tp=size(I,2);

if(nargin>3)
    x=pixel(1);
    y=pixel(2);
    
    Ii=reshape(I(x,y,:),[tp 1]);
else
   Ii=I'; 
end

D=diff(b,3,1);%%third order finite difference operator order 3 matrix can be constructed as d=[-1 2 -1 0 0 0];  M=toeplitz([A(1) fliplr(A(2:end))], A)delete last two rows and last col
%%the non-neg lagrangian dual way of NNLS
C=chol(inv(V'*V));

Cons=2*C*V'*Ii;
Coeff=C*D';

 alpha = lsqnonnegvect(Coeff,Cons); %% non negative least squares
% 
% ind=(find(alpha>0));
% alpha(find(alpha>0));
 c=inv(V'*V)*(V'*Ii-D'*alpha)
 

% 
% %%regular least squares
 c1= (V'*V)^-1*V'*Ii 


%%constrained least squares, identical to the dual method
%c2=lsqlin(V,Ii,D,zeros(size(D,1),1))
%  [c c1 c2]
% 
%  e=(Ii-V*c2).*(Ii-V*c2);
%  [Ii V*c2]
%  error=sum(double(e))
if(doplot)
%     plot(1:256,Ii,'r-');
%     pause
%     figure
    plot(1:256,Ii,'r-',1:256,V*c,'b-',1:256,V*c1,'g-',1:256,V*c2,'m');
    legend('fitted function convolved with irf','Fitting with LSE using Laguerre bases(dual)','Fitting with LSE using Laguerre bases','Fitting with LSE using Laguerre bases(lsqlin)');
end
%         pause
%
end