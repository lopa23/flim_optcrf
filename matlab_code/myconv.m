function convA=myconv(A,b)

m=numel(b);
hm=floor(m/2);

paddedA=[zeros(1,hm) A zeros(1,hm)];

convA=[];
j=1;
for i=hm+1:size(paddedA,2)-hm
    if(mod(m,2)>0)
        convA(j)=sum(paddedA(i-hm:i+hm).*b);
    else
       convA(j)=sum(paddedA(i-hm:i+hm-1).*b);
    end
    j=j+1;
end
%%convA