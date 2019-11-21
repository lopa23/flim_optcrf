function V=myconv(A,b,tp,L)
% if(size(A,1)>size(A,2));
%     A=A'
%     
% end
m=numel(b);
hm=floor(m/2);


paddedA=[zeros(L,hm) A' zeros(L,hm)];

convA=[];
j=1;
V=zeros(tp,L);
for k=hm+1:size(paddedA,2)-hm
   for l=1:L
      if(mod(m,2)>0)
            V(k-hm,l)= sum(paddedA(l,k-hm:k+hm).*b);
             
       
      else
           V(k-hm,l)= sum(paddedA(l,k-hm:k+hm-1).*b);
          
      end
    
   end
     
end 