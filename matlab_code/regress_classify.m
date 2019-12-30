function regress_classify(b,V,I,Label)

doplot=1;
tp=size(I,2);
n=size(b,2);
m=size(I,1);
Ii=I'; 
D=diff(b,3,1);

H=V'*V; %*2 if using quadprog


f=-2*Ii'*V;
% size(f)



%c1 = quadprog(2*H,f',D,zeros(size(D,1),1));
%c1=solve_using_cvx(H,f,D,zeros(size(D,1),size(f,1)),V,Ii);
c=Generate_QPparams(H,f,[],[],D,zeros(size(D,1),size(f,1)),m,Label);
c=reshape(c,[n m]);

%[c c1]

e=(Ii-V*c).*(Ii-V*c);
%  [Ii V*c2]
  error=sum(double(e));

if(doplot)
%     plot(1:256,Ii,'r-');
%     pause
%     figure
    plot(1:256,Ii,'r-',1:256,V*c,'b-');
    legend('fitted function convolved with irf','Fitting with LSE using Laguerre bases(quadprog)');
end