% LS-OMP
function [lsomp_record,flop_lsomp] = LSOMP(A,signal_input,error_thres)
m = size(A,1);
n = size(A,2);
lsomp_record_po = zeros(size(signal_input,2),1);
lsomp_record = zeros(n, size(signal_input,2));
lsomp_err = ones(size(signal_input,2),1);
lsomp_record_itr = zeros(size(signal_input,2),1);
lsomp_itr = 0;
lsomp_signal = signal_input;
flop_lsomp = 0;
while max(lsomp_err > error_thres)==1
    lsomp_itr = lsomp_itr + 1;
    for signal_N = 1:size(signal_input,2)
        if lsomp_err(signal_N) > error_thres
            if lsomp_itr == 1
                inner_product = (A'*signal_input(:,signal_N));
                [v_,po]=max(abs(inner_product),[],1);
                flop_lsomp = flop_lsomp + 2*m*n;
            else              
                lsuse_atom = A(:,lsomp_record_po(signal_N,1:lsomp_itr-1));
                M_inv = (lsuse_atom'*lsuse_atom)^(-1);
                flop_lsomp = flop_lsomp + 2*2*m*size(lsuse_atom,2)*size(lsuse_atom,2);
                b = signal_input(:,signal_N);
                test_error = [];
                for test_i = 1:size(A,2)
                    if min(abs(test_i-lsomp_record_po(signal_N,:))) == 0
                        test_error = [test_error 10000];
                    else
                        ai = A(:,test_i);
                        c = lsuse_atom'*ai;
                        %matr = [lsuse_atom'*lsuse_atom c;c' 1];
                        ls_mat = [M_inv+M_inv*c*c'*M_inv -M_inv*c; -c'*M_inv 1];
                        res = ls_mat*[lsuse_atom'*b; ai'*b];
                        flop_lsomp = flop_lsomp + 2*m*(size(lsuse_atom,2)+1)*(size(lsuse_atom,2)+1);
                        test_error = [test_error norm([lsuse_atom ai]*res - b)];
                    end
                end
                [v_, po] = min(abs(test_error),[],2);
            end    
            
            lsomp_record_po(signal_N,lsomp_itr) = po;
            use_atom = A(:,lsomp_record_po(signal_N,:));
            use_atom_val = (use_atom'*use_atom)^(-1)*use_atom'*signal_input(:,signal_N);
            flop_lsomp = flop_lsomp + 2*2*m*size(use_atom,2)*size(use_atom,2);
            cons_sig = use_atom*use_atom_val;
            for atom_num = 1:size(use_atom,2)
                lsomp_record(lsomp_record_po(signal_N,atom_num),signal_N) = use_atom_val(atom_num);
            end
            lsomp_signal(:,signal_N) = signal_input(:,signal_N) - cons_sig;
            lsomp_err(signal_N) = norm(lsomp_signal(:,signal_N));
            lsomp_record_itr(signal_N) = lsomp_itr;
        end
    end
end
flop_lsomp = flop_lsomp/size(signal_input,2);