function [omp_record,flop_omp] = OMP(A,signal_input,error_thres)
m = size(A,1);
n = size(A,2);
omp_record_po = zeros(size(signal_input,2),1);
omp_record = zeros(n, size(signal_input,2));
omp_err = ones(size(signal_input,2),1);
omp_record_itr = zeros(size(signal_input,2),1);
omp_itr = 0;
flop_omp = 0;
omp_signal = signal_input;
while max(omp_err > error_thres)==1
    omp_itr = omp_itr + 1;
    inner_product = (A'*omp_signal);
    [v_,po]=max(abs(inner_product),[],1);
    
    for signal_N = 1:size(signal_input,2)
        if omp_err(signal_N) > error_thres
            flop_omp = flop_omp + 2*m*n;
            omp_record_po(signal_N,omp_itr) = po(signal_N);
            use_atom = A(:,omp_record_po(signal_N,:));
            use_atom_val = (use_atom'*use_atom)^(-1)*use_atom'*signal_input(:,signal_N);
            flop_omp = flop_omp + 2*2*m*size(use_atom,2)*size(use_atom,2)+size(use_atom,2)*m*(2*size(use_atom,2)-1);
            
            cons_sig = use_atom*use_atom_val;
            for atom_num = 1:size(use_atom,2)
                omp_record(omp_record_po(signal_N,atom_num),signal_N) = use_atom_val(atom_num);
            end
            omp_signal(:,signal_N) = signal_input(:,signal_N) - cons_sig;
            omp_err(signal_N) = norm(omp_signal(:,signal_N));
            omp_record_itr(signal_N) = omp_itr;
        end
    end
end
flop_omp = flop_omp/size(signal_input,2);