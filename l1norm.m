function [result, err, flop_l1] = l1norm(A, b, lambda, error, step, itr_limit)
m = size(A,1);
n = size(A,2);
x = (A'*A+0.00001*eye(n))^-1*A'*b;
res = norm(b - A*x);
res_rec = res;
itr = 0;
flop_l1 = 0;
while itr < itr_limit
    djdx = 2*A'*A*x - 2*A'*b + lambda*sign(x);
    flop_l1 = flop_l1 + 2*2*n*m + 2*n*m  + 2*n;
    x = x - step*djdx;
    res = norm(b - A*x);
    res_rec = [res_rec, res];
%     if res < error
%         break;
%     end
%     if mod(itr,1000)==0
%         fprintf('This is iteration %f, error %f\n', itr,res);
%     end
    itr = itr + 1;
end
result = x;
err = res;
% figure
% plot(res_rec);
% hold on;
% plot([0,itr], [error, error]);
