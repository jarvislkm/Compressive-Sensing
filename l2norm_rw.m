function [result, err, flop] = l2norm_rw(A, b, lambda, error, itr_limit)
p = 0.9;
m = size(A,1);
n = size(A,2);
x = (A'*A+0.001*eye(n))^-1*A'*b;
res = norm(b - A*x);
res_rec = res;
flop = 0;
itr = 0;
while itr < itr_limit
    w = abs(x).^(1-p/2).*eye(n);
    flop = flop +2*n;
    x = w*w*A'*(A*w*w*A'+lambda.*eye(m))^-1*b;
    flop = flop + 3*n*m*2*n + n*m*2*m + m*m + n*2*m;
    res = norm(b - A*x);
    res_rec = [res_rec, res];
%     if res < error
%        break;
%     end
%     if mod(itr,100)==0
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
