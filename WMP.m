%This is Weak-MP algorighm
function [wmp_record, flop_wmp] = WMP(A,signal_input,t0,error_thres)
t = t0;
m = size(A,1);
n = size(A,2);
wmp_record = zeros(n, size(signal_input,2));
wmp_err = ones(size(signal_input,2),1);
wmp_record_itr = zeros(size(signal_input,2),1);
wmp_itr = 0;
wmp_signal = signal_input;
flop_wmp = 0;
while max(wmp_err > error_thres)==1
    wmp_itr = wmp_itr + 1;
    norm_sig = t*vecnorm(wmp_signal,2,1);
    inner_product = abs(A'*wmp_signal) - repmat(norm_sig,n,1);
    po = [];
    for count_inner = 1:size(inner_product,2)
        po_a = find(inner_product(:,count_inner)>0, 1, 'first');
        if isempty(po_a)
            [v__, po_a] = max(inner_product(:,count_inner));
        end
        po = [po po_a];
    end
    
    for signal_N = 1:size(signal_input,2)
        if wmp_err(signal_N) > error_thres
            flop_wmp = flop_wmp + po(signal_N)*2*m;
            v = A(:,po(signal_N))'*wmp_signal(:,signal_N);      
            wmp_record(po(signal_N),signal_N) = wmp_record(po(signal_N),signal_N)+v;
            wmp_signal(:,signal_N) = wmp_signal(:,signal_N) - v*A(:,po(signal_N));
            wmp_err(signal_N) = norm(wmp_signal(:,signal_N));
            wmp_record_itr(signal_N) = wmp_itr;
        end
    end
end
flop_wmp = flop_wmp/size(signal_input,2);