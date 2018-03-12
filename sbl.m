function [result, err, flop] = sbl(A, b)
n = size(A,2);
m = size(A,1);
gamma = (ones(n,1)).^2.*eye(n)/sqrt(n);
itr = 0;
itr_limit = 50;
res_rec = [];
res_min = 1;
flop = 0;
while itr < itr_limit
    sigma_yy = A*gamma*A'+ 0.00001*eye(m);
    sigma_yy_inv = sigma_yy^-1;
    flop = flop + 2*m*n*(m+n +1);
    miu_x = gamma*A'*sigma_yy_inv*b;
    flop = flop + 2*n*m*(m+n +1);
    cov_x = gamma - gamma*A'* sigma_yy_inv*A*gamma;
    flop = flop + 2*n*m*(2*m+2*n +1);
    gamma = (miu_x.^2).*eye(n) + eye(n).*cov_x;
    flop = flop + 4*n;
    res = norm(b-A*miu_x);
    res_rec = [res_rec res];
    if res < res_min
       res_min = res;
       x_rec = miu_x;
    end
%     if res < error
%         x_rec = miu_x;
%         break;
%     end
%     if mod(itr,1000)==0
%         fprintf('This is iteration %f, error %f\n', itr,res);
%     end
    itr = itr + 1;
end
result = miu_x;
err = res_min;
% figure
% plot(res_rec);
% hold on;
