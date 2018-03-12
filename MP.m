function [mp_record,flop_mp] = MP(A,signal_input,error_thres)
m = size(A,1);
n = size(A,2);
mp_record = zeros(n, size(signal_input,2));
mp_err = ones(size(signal_input,2),1);
mp_record_itr = zeros(size(signal_input,2),1);
mp_itr = 0;
mp_signal = signal_input;
flop_mp = 0;
while max(mp_err > error_thres)==1
    mp_itr = mp_itr + 1;
    inner_product = (A'*mp_signal);
    
    [v_,po]=max(abs(inner_product),[],1);
    
    for signal_N = 1:size(signal_input,2)
        if mp_err(signal_N) > error_thres
            flop_mp = flop_mp + 2*m*n;
            v = A(:,po(signal_N))'*mp_signal(:,signal_N);      
            mp_record(po(signal_N),signal_N) = mp_record(po(signal_N),signal_N)+v;
            flop_mp = flop_mp + n;
            mp_signal(:,signal_N) = mp_signal(:,signal_N) - v*A(:,po(signal_N));
            flop_mp = flop_mp + 2*n;
            mp_err(signal_N) = norm(mp_signal(:,signal_N));
            flop_mp = flop_mp + 2*n;
            mp_record_itr(signal_N) = mp_itr;
        end
    end
end
flop_mp = flop_mp/size(signal_input,2);