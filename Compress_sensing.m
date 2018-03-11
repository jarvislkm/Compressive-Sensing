%% Programming Problem
clear all;
A = randn(30,50);
A = S_normc_self(A);
error_thres = 1e-4;

cardinality = 15;
size_set = 100;
signal = [];

for i = 1:cardinality
    for s = 1:size_set
        position = randperm(50,i);
        this = zeros(50,1);
        this(position,1) = 1;
        signal = [signal this];
    end
end
%non_zero = tan(pi*(rand(size(signal,1), size(signal,2))-1/2));
%non_zero = randn(size(signal,1), size(signal,2));
%sign = 2*randi([0,1],size(signal,1), size(signal,2))-1;

%% normal dist
non_zero = randn(size(signal,1), size(signal,2));
% %% Cauchy
% non_zero = tan(pi*(rand(size(signal,1), size(signal,2))-1/2));
% %% {-1,1} binomal
% non_zero = 2*randi([0,1],size(signal,1), size(signal,2))-1;
% %% [-2,-1] U [1,2]
% non_zero = rand(size(signal,1), size(signal,2))+1;
% sign = 2*randi([0,1],size(signal,1), size(signal,2))-1;
% non_zero = non_zero.*sign;

%%
%non_zero = non_zero.*signal;
non_zero = signal;
signal_input = A*(non_zero);

%% WMP
tic
t = 0.5;
wmp_record = zeros(50, size(signal_input,2));
wmp_err = ones(size(signal_input,2),1);
wmp_record_itr = zeros(size(signal_input,2),1);
wmp_itr = 0;
wmp_signal = signal_input;
flop_wmp = 0;
while max(wmp_err > error_thres)==1
    wmp_itr = wmp_itr + 1;
    norm_sig = t*vecnorm(wmp_signal,2,1);
    inner_product = abs(A'*wmp_signal) - repmat(norm_sig,50,1);
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
            flop_wmp = flop_wmp + po(signal_N)*59;
            v = A(:,po(signal_N))'*wmp_signal(:,signal_N);      
            wmp_record(po(signal_N),signal_N) = wmp_record(po(signal_N),signal_N)+v;
            wmp_signal(:,signal_N) = wmp_signal(:,signal_N) - v*A(:,po(signal_N));
            wmp_err(signal_N) = norm(wmp_signal(:,signal_N));
            wmp_record_itr(signal_N) = wmp_itr;
        end
    end
end
toc
result_len_w = vecnorm(non_zero-wmp_record)./vecnorm(non_zero);
result_len_w = reshape(result_len_w,size_set,size(result_len_w,2)/size_set);
result_len_w = mean(result_len_w);

s1_w = vecnorm(non_zero);
s2_w = vecnorm(wmp_record);
s3_w = vecnorm(signal.*wmp_record);
s_w = (max(s1_w,s2_w)-s3_w)./max(s1_w,s2_w);
s_w = reshape(s_w,size_set,size(s_w,2)/size_set);
result_dis_w = mean(s_w);
%% LS-OMP
tic
lsomp_record_po = zeros(size(signal_input,2),1);
lsomp_record = zeros(50, size(signal_input,2));
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
                flop_lsomp = flop_lsomp + 59*50;
            else              
                lsuse_atom = A(:,lsomp_record_po(signal_N,1:lsomp_itr-1));
                M_inv = (lsuse_atom'*lsuse_atom)^(-1);
                flop_lsomp = flop_lsomp + 2*59*size(lsuse_atom,2)*size(lsuse_atom,2);
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
                        flop_lsomp = flop_lsomp + 61*(size(lsuse_atom,2)+1)*(size(lsuse_atom,2)+1);
                        test_error = [test_error norm([lsuse_atom ai]*res - b)];
                    end
                end
                [v_, po] = min(abs(test_error),[],2);
            end    
            
            lsomp_record_po(signal_N,lsomp_itr) = po;
            use_atom = A(:,lsomp_record_po(signal_N,:));
            use_atom_val = (use_atom'*use_atom)^(-1)*use_atom'*signal_input(:,signal_N);
            flop_lsomp = flop_lsomp + 2*59*size(use_atom,2)*size(use_atom,2);
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
toc

result_len_ls = vecnorm(non_zero-lsomp_record)./vecnorm(non_zero);
result_len_ls = reshape(result_len_ls,size_set,size(result_len_ls,2)/size_set);
result_len_ls = mean(result_len_ls);

s1_ls = vecnorm(non_zero);
s2_ls = vecnorm(lsomp_record);
s3_ls = vecnorm(signal.*lsomp_record);
s_ls = (max(s1_ls,s2_ls)-s3_ls)./max(s1_ls,s2_ls);
s_ls = reshape(s_ls,size_set,size(s_ls,2)/size_set);
result_dis_ls = mean(s_ls);

%% OMP
tic
omp_record_po = zeros(size(signal_input,2),1);
omp_record = zeros(50, size(signal_input,2));
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
            flop_omp = flop_omp + 59*50;
            omp_record_po(signal_N,omp_itr) = po(signal_N);
            use_atom = A(:,omp_record_po(signal_N,:));
            use_atom_val = (use_atom'*use_atom)^(-1)*use_atom'*signal_input(:,signal_N);
            flop_omp = flop_omp + 2*59*size(use_atom,2)*size(use_atom,2)+size(use_atom,2)*30*(2*size(use_atom,2)-1);
            
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
toc

result_len_o = vecnorm(non_zero-omp_record)./vecnorm(non_zero);
result_len_o = reshape(result_len_o,size_set,size(result_len_o,2)/size_set);
result_len_o = mean(result_len_o);

s1_lo = vecnorm(non_zero);
s2_lo = vecnorm(omp_record);
s3_lo = vecnorm(signal.*omp_record);
s_o = (max(s1_lo,s2_lo)-s3_lo)./max(s1_lo,s2_lo);
s_o = reshape(s_o,size_set,size(s_o,2)/size_set);
result_dis_o = mean(s_o);

%% MP
tic
mp_record = zeros(50, size(signal_input,2));
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
            flop_mp = flop_mp + 59*50;
            v = A(:,po(signal_N))'*mp_signal(:,signal_N);      
            mp_record(po(signal_N),signal_N) = mp_record(po(signal_N),signal_N)+v;
            flop_mp = flop_mp + 50;
            mp_signal(:,signal_N) = mp_signal(:,signal_N) - v*A(:,po(signal_N));
            flop_mp = flop_mp + 100;
            mp_err(signal_N) = norm(mp_signal(:,signal_N));
            flop_mp = flop_mp + 100;
            mp_record_itr(signal_N) = mp_itr;
        end
    end
end
toc

result_len_mp = vecnorm(non_zero-mp_record)./vecnorm(non_zero);
result_len_mp = reshape(result_len_mp,size_set,size(result_len_mp,2)/size_set);
result_len_mp = mean(result_len_mp);

s1_mp = vecnorm(non_zero);
s2_mp = vecnorm(mp_record);
s3_mp = vecnorm(signal.*mp_record);
s_mp = (max(s1_mp,s2_mp)-s3_mp)./max(s1_mp,s2_mp);
s_mp = reshape(s_mp,size_set,size(s_mp,2)/size_set);
result_dis_mp = mean(s_mp);

%% Threshold
tic
k = 10;
tmp_record = zeros(50, size(signal_input,2));
tmp_err = ones(size(signal_input,2),1);
tmp_record_itr = zeros(size(signal_input,2),1);
tmp_itr = 0;
tmp_signal = signal_input;
flop_th = 0;
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
            flop_th = flop_th + 59*50;
            
            use_atom = A(:,po(signal_N,:));
            use_atom_val = (use_atom'*use_atom)^(-1)*use_atom'*tmp_signal(:,signal_N);
            flop_th = flop_th + 59*100+300*19+10*59;
            
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
toc    

result_len_t = vecnorm(non_zero-tmp_record)./vecnorm(non_zero);
result_len_t = reshape(result_len_t,size_set,size(result_len_t,2)/size_set);
result_len_t = mean(result_len_t);

s1_t = vecnorm(non_zero);
s2_t = vecnorm(tmp_record);
s3_t = vecnorm(signal.*tmp_record);
s_t = (max(s1_t,s2_t)-s3_t)./max(s1_t,s2_t);
s_t = reshape(s_t,size_set,size(s_t,2)/size_set);
result_dis_t = mean(s_t);


%% self l1norm 
tic
l1_record = zeros(50, size(signal_input,2));
for count = 1:size(signal_input,2)
    lambda = 0.005;
    [recon,status]=l1norm(A,signal_input(:,count),lambda,error_thres,0.1);
    l1_record(:,count) = recon;
end
result_len_l1 = vecnorm(non_zero-l1_record)./vecnorm(non_zero);
result_len_l1 = reshape(result_len_l1,size_set,size(result_len_l1,2)/size_set);
result_len_l1 = mean(result_len_l1);

s1_l = vecnorm(non_zero);
s2_l = vecnorm(l1_record);
s3_l = vecnorm(signal.*l1_record);
s_l = (max(s1_l,s2_l)-s3_l)./max(s1_l,s2_l);
s_l = reshape(s_l,size_set,size(s_l,2)/size_set);
result_dis_l1 = mean(s_l);

toc
%% self reweight-l2 
tic
l2_record = zeros(50, size(signal_input,2));
for count = 1:size(signal_input,2)
    lambda = 0.001;
    [recon,status]=l2norm_rw(A,signal_input(:,count),lambda,error_thres);
    l2_record(:,count) = recon;
end
result_len_l2 = vecnorm(non_zero-l2_record)./vecnorm(non_zero);
result_len_l2 = reshape(result_len_l2,size_set,size(result_len_l2,2)/size_set);
result_len_l2 = mean(result_len_l2);

s1_l2 = vecnorm(non_zero);
s2_l2 = vecnorm(l2_record);
s3_l2 = vecnorm(signal.*l2_record);
s_l2 = (max(s1_l2,s2_l2)-s3_l2)./max(s1_l2,s2_l2);
s_l2 = reshape(s_l2,size_set,size(s_l2,2)/size_set);
result_dis_l2 = mean(s_l2);

toc
%% self reweight-l1
tic
l1rw_record = zeros(50, size(signal_input,2));
for count = 1:size(signal_input,2);
    lambda = 0.004;
    [recon,status]=l1norm_rw(A,signal_input(:,count),lambda,error_thres,0.05);
    l1rw_record(:,count) = recon;
end
result_len_l1rw = vecnorm(non_zero-l1rw_record)./vecnorm(non_zero);
result_len_l1rw = reshape(result_len_l1rw,size_set,size(result_len_l1rw,2)/size_set);
result_len_l1rw = mean(result_len_l1rw);

s1_l1rw = vecnorm(non_zero);
s2_l1rw = vecnorm(l1rw_record);
s3_l1rw = vecnorm(signal.*l1rw_record);
s_l1rw = (max(s1_l1rw,s2_l1rw)-s3_l1rw)./max(s1_l1rw,s2_l1rw);
s_l1rw = reshape(s_l1rw,size_set,size(s_l1rw,2)/size_set);
result_dis_l1rw = mean(s_l1rw);

toc
%% self sbl
tic
sbl_record = zeros(50, size(signal_input,2));
for count = 1:size(signal_input,2)
    [recon,status]=sbl(A,signal_input(:,count));
    sbl_record(:,count) = recon;
end
result_len_sbl = vecnorm(non_zero-sbl_record)./vecnorm(non_zero);
result_len_sbl = reshape(result_len_sbl,size_set,size(result_len_sbl,2)/size_set);
result_len_sbl = mean(result_len_sbl);

s1_sbl = vecnorm(non_zero);
s2_sbl = vecnorm(sbl_record);
s3_sbl = vecnorm(signal.*sbl_record);
s_sbl = (max(s1_sbl,s2_sbl)-s3_sbl)./max(s1_sbl,s2_sbl);
s_sbl = reshape(s_sbl,size_set,size(s_sbl,2)/size_set);
result_dis_sbl = mean(s_sbl);

toc
%% plot
figure
length = size(result_len_mp,2);

plot(result_len_mp);hold on;
plot(result_len_o);hold on;
plot(result_len_ls);hold on;
plot(result_len_w);hold on;
plot(result_len_t);hold on;
plot(result_len_l1);hold on;
plot(result_len_l2);hold on;
plot(result_len_l1rw);hold on;
plot(result_len_sbl);
xlim([1 length]);
legend("MP","OMP","LS-OMP","W-MP","Threshold","l1-norm","l2rw-norm","l1rw-norm","sbl");
title("L_2 Error of different method, [-2,-1]U[1,2]");
xlabel("Cardinality of the true solution");
ylabel("Average and Relative L_2 Error")
figure
plot(result_dis_mp);hold on;
plot(result_dis_o);hold on;
plot(result_dis_ls);hold on;
plot(result_dis_w);hold on;
plot(result_dis_t);hold on;
plot(result_dis_l1);hold on;
plot(result_dis_l2);hold on;
plot(result_dis_l1rw);hold on;
plot(result_dis_sbl);
xlim([1 length]);
legend("MP","OMP","LS-OMP","W-MP","Threshold","l1-norm","l2rw-norm","l1rw-norm","sbl");
title("Distance of different method, [-2,-1]U[1,2]");
xlabel("Cardinality of the true solution");
ylabel("Probability of Error in Support")

%% 
% m_h_s = [];
% for count = 1:4000
%     m_h = 0;
%     A = randn(30,100);
%     A = S_normc_self(A);
%     for i = 1:size(A,2)-1
%         for j = i+1:size(A,2)
%             m_h_now = abs(A(:,i)'*A(:,j));
%             if m_h_now > m_h
%                 m_h = m_h_now;
%             end
%         end
%     end
%     m_h_s = [m_h_s m_h];
%     
% end
% bound = (1+1./m_h_s)/2;
% figure
% hist(bound,20);
% title("Distribution of bound");
% xlabel("bound")
% ylabel("frequency")
