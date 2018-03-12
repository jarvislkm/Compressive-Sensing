function [tmp_record,flop_tmp] = TMP(A,signal_input,k0,error_thres)
k = k0;
m = size(A,1);
n = size(A,2);
tmp_record = zeros(n, size(signal_input,2));
tmp_err = ones(size(signal_input,2),1);
tmp_record_itr = zeros(size(signal_input,2),1);
tmp_itr = 0;
tmp_signal = signal_input;
flop_tmp = 0;
while max(tmp_err > error_thres)==1
    tmp_itr = tmp_itr + 1;
    inner_product = (A'*tmp_signal);
    po = [];
    for k_num = 1:k
        [v_,p_rec]=max(abs(inner_product),[],1);
        for s_count = 1:size(inner_product,2)
            inner_product(p_rec(s_count),s_count) = 0;
        end
        po = [po p_rec'];
    end
    
    for signal_N = 1:size(signal_input,2)
        if tmp_err(signal_N) > error_thres
            flop_tmp = flop_tmp + 2*m*n;
            
            use_atom = A(:,po(signal_N,:));
            use_atom_val = (use_atom'*use_atom)^(-1)*use_atom'*tmp_signal(:,signal_N);
            flop_tmp = flop_tmp + 2*m*2*n + k*m*2*k + k*2*m;
            
            cons_sig = use_atom*use_atom_val;
            for record_count = 1:k
                tmp_record(po(signal_N,record_count),signal_N) = tmp_record(po(signal_N,record_count),signal_N)+use_atom_val(record_count);
            end
            tmp_signal(:,signal_N) = tmp_signal(:,signal_N) - cons_sig;
            tmp_err(signal_N) = norm(tmp_signal(:,signal_N));
            tmp_record_itr(signal_N) = tmp_itr;
        end
    end    
end
flop_tmp = flop_tmp/size(signal_input,2);