function [blm_dmu,new_dmu_X,new_dmu_Y]=blmsf(X,Y,d1,d2,d3,bh) 
M=10^6;
wc=10^-5;
n=length(X(1,:));
new_dmu_X=[];
new_dmu_Y=[];

% model(3)
k=binvar(n,1,'full');
theta=sdpvar(1,1);
lam=sdpvar(n+1,1);
Constraints=[];
Constraints=[Constraints,X*lam(1:n,1)<=theta*X(:,bh)];
Constraints=[Constraints,Y*lam(1:n,1)>=Y(:,bh)];
Constraints=[Constraints,d1*(sum(lam(1:n))+d2*(-1)^d3*lam(n+1))==d1];
Constraints=[Constraints,lam>=0];
Constraints=[Constraints,lam(1:n,1)<=M*k];
Objective=theta+wc*10*sum(k);


% Set some options for YALMIP and solver8
options = sdpsettings('verbose',1,'solver','mosek');
% Solve the problem
sol = optimize(Constraints,Objective,options);
% Analyze error flags
if sol.problem == 0
 % Extract and display value
 k_opt = value(k);    % selection of DMU
 zyj = value(Objective);
 xl(1)=value(theta);  % efficiency
 combination_coeff=value(lam);
else
 display('Hmm, something went wrong!');
 sol.info
 yalmiperror(sol.problem)
end

new_dmu_X = vertcat(new_dmu_X, (X*combination_coeff(1:n,1))');
new_dmu_Y = vertcat(new_dmu_Y, (Y*combination_coeff(1:n,1))');

% blm_dmu = zeros(n, n);
sum0=1;  % the layer of envelope
sum1=0;  % the index of dmu in the same layer
for j=1:n
    if k_opt(j)>1-wc
        sum1=sum1+1;
        blm_dmu(sum0,sum1)=j;   % record the index of DMU which lamda > 0 
    end
end



while sum1>1|xl(sum0)<1-wc
   
   % model(4)
   k=binvar(n,1,'full');
   theta=sdpvar(1,1);
   lam=sdpvar(n+1,1);
   Constraints=[];
   Constraints=[Constraints,X*lam(1:n,1)<=theta*X(:,bh)];
   Constraints=[Constraints,Y*lam(1:n,1)>=Y(:,bh)];
   Constraints=[Constraints,d1*(sum(lam(1:n))+d2*(-1)^d3*lam(n+1))==d1];
   Constraints=[Constraints,lam>=0];
   Constraints=[Constraints,lam(1:n,1)<=M*k];
   %Constraints=[Constraints,k(bh)==0];
   Constraints=[Constraints,theta>=xl(sum0)+wc];  % efficiency not lower than before
   for i=1:sum0                                   % traversal each layer 
       xs=zeros(1,n); 
       for j=1:sum(blm_dmu(i,:)~=0)
           xs(blm_dmu(i,j))=1;
       end
       Constraints=[Constraints,xs*k<=sum(blm_dmu(i,:)~=0)-1];
   end
   Objective=theta+wc*10*sum(k);
   
   % Set some options for YALMIP and solver8
   options = sdpsettings('verbose',1,'solver','mosek');
   % Solve the problem
   sol = optimize(Constraints,Objective,options);
   % Analyze error flags
   if sol.problem == 0
     % Extract and display value
     k_opt = value(k);
     zyj = value(Objective);
     sum0=sum0+1;               % next layer
     xl(sum0)=value(theta);
     combination_coeff=value(lam);
   else
     display('Hmm, something went wrong!');
     sol.info
     yalmiperror(sol.problem)
   end

   new_dmu_X = vertcat(new_dmu_X, (X*combination_coeff(1:n,1))');
   new_dmu_Y = vertcat(new_dmu_Y, (Y*combination_coeff(1:n,1))');
   sum1=0;                      % record the index in this layer
   for j=1:n
    if k_opt(j)>1-wc
        sum1=sum1+1;
        blm_dmu(sum0,sum1)=j;
    end
   end
end

new_dmu_X = new_dmu_X(1:end-1,:);
new_dmu_Y = new_dmu_Y(1:end-1,:);

blm_dmu=[xl(1:end-1)' blm_dmu(1:end-1,:)];   
% for i=1:length(blm_dmu(:,1))
%     for j=1:sum(blm_dmu(i,:)~=0)-1
%         blsd(i,blm_dmu(i,j+1))=(1-blm_dmu(i,1))/(sum(blm_dmu(i,:)~=0)-1);
%     end
% end
% if length(blsd(1,:))<n
%     blsd=[blsd zeros(length(blsd(:,1)),n-length(blsd(1,:)))];
% end
blm_dmu = [bh*ones(sum0-1,1)  blm_dmu];

end

