%% Compressive Sensing Algorithms
clear all;
error_thres = 1e-4;

% Support distribution:
% "1:Normal",
% "2:Cauchy",
% "3:Binomial {+1,-1}",
% "4:Uniform Distribution [-2,-1] U [1,2]"
support_distribution = 3;

cardinality = 15; % Max test L0-norm
size_set = 10;   % set of example for each L0-norm
m = 30;           % Measurement times 
n = 50;           % Signal length

% define sensing matrix
A = randn(m,n);
A = S_normc_self(A);

%Signal generation
signal = [];      
for i = 1:cardinality
    for s = 1:size_set
        position = randperm(n,i);
        this = zeros(n,1);
        this(position,1) = 1;
        signal = [signal this];
    end
end

if support_distribution == 1
    disp("Using Normal Distribution N(0,1) for support vector");
    non_zero = randn(size(signal,1), size(signal,2));
elseif support_distribution == 2
    disp("Using Cauchy Distribution C(0,1)for support vector");
    non_zero = tan(pi*(rand(size(signal,1), size(signal,2))-1/2));
elseif support_distribution == 3
    disp("Using Binomial Distribution {+1, -1} for support vector");
    non_zero = 2*randi([0,1],size(signal,1), size(signal,2))-1;
elseif support_distribution == 4
    disp("Using Uniform Distribution [-2,-1] U [1,2] for support vector");
    non_zero = rand(size(signal,1), size(signal,2))+1;
    sign = 2*randi([0,1],size(signal,1), size(signal,2))-1;
    non_zero = non_zero.*sign;
else
    disp("This is not available");
    quit
end
    
non_zero = non_zero.*signal;
signal_input = A*(non_zero);

%% WMP
disp("Weak-MP is called");

tic
t0 = 0.5;
[wmp_record, flop_wmp] = WMP(A,signal_input,t0,error_thres);    
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
disp("LS-OMP is called");

tic
[lsomp_record, flop_lsomp] = LSOMP(A,signal_input,error_thres); 
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
disp("OMP is called");
tic
[omp_record,flop_omp] = OMP(A,signal_input,error_thres);
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
disp("MP is called");
tic
[mp_record,flop_mp] = MP(A,signal_input,error_thres);
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
disp("Threshold is called");
tic
k0 = 10;
[tmp_record,flop_tmp] = TMP(A,signal_input,k0,error_thres);
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
disp("L1-norm is called");
tic
l1_record = zeros(n, size(signal_input,2));
itr_limit = 10000;
flop_l1 = [];
for count = 1:size(signal_input,2)
    lambda = 0.04;
    [recon, err, flop]=l1norm(A,signal_input(:,count),lambda,error_thres,0.1,itr_limit);
    flop_l1 = [flop_l1, flop];
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
flop_l1 = mean(flop_l1);
toc
%% self reweight-l2 
disp("Reweighted L2-norm is called");
tic
l2_record = zeros(n, size(signal_input,2));
itr_limit = 100;
flop_l2 = [];
for count = 1:size(signal_input,2)
    lambda = 0.001;
    [recon, error, flop]=l2norm_rw(A,signal_input(:,count),lambda,error_thres,itr_limit);
    flop_l2 = [flop_l2, flop];
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
flop_l2 = mean(flop_l2);
toc
%% self reweight-l1
disp("Reweighted L1-norm is called");
tic
l1rw_record = zeros(n, size(signal_input,2));
flop_l1rw = [];
for count = 1:size(signal_input,2)
    lambda = 0.01;
    [recon,error, flop]=l1norm_rw(A,signal_input(:,count),lambda,error_thres,0.1);
    flop_l1rw = [flop_l1rw, flop];
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
flop_l1rw = mean(flop_l1rw);
toc
%% self sbl
disp("Sparse Bayes Learning is called");
tic
flop_sbl = [];
sbl_record = zeros(n, size(signal_input,2));
for count = 1:size(signal_input,2)
    [recon,err, flop]=sbl(A,signal_input(:,count));
    sbl_record(:,count) = recon;
    flop_sbl = [flop_sbl, flop];
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
flop_sbl = mean(flop_sbl);
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
legend("MP","OMP","LS-OMP","W-MP","Threshold","l1-norm",...
    "l2rw-norm","l1rw-norm","sbl",'Location','northwest');
title("L_2 Error of different method, Cauchy Distribution");
xlabel("Cardinality of the true solution");
ylabel("Average and Relative L_2 Error")
x0=10;
y0=10;
width=550;
height=400;
set(gcf,'units','points','position',[x0,y0,width,height])

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
legend("MP","OMP","LS-OMP","W-MP","Threshold","l1-norm",...
    "l2rw-norm","l1rw-norm","sbl",'Location','northwest');
title("Distance of different method, Cauchy Distribution");
xlabel("Cardinality of the true solution");
ylabel("Probability of Error in Support")
x0=10;
y0=10;
width=550;
height=400;
set(gcf,'units','points','position',[x0,y0,width,height])
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
