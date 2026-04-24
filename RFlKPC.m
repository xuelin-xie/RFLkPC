function cluster_output = RFlKPC(data,U_ini,K,m,alpha,lambda,weight)
% Robust fuzzy local K-plane clustering
% data:N*D
% U_ini:N*K, membership matrix, initialized with KPC's U result
% weight:N*1
% 
% Stopping tolerance
threshold = 1e-4; 
% Get number of samples N, data dimension D
[N,D] = size(data);
data = double(data);
if nargin<7
    weight = ones(N,1);
end

%% initial values for hyperplanes' parameters
[Vk,muk] = inial_Vk_muk(data,U_ini,K,D,weight);
D_ik = error_ik(Vk,muk,data,N,K,alpha,lambda,weight);
U_ik = fuzzy_u_ik(D_ik,N,K,m);
%% Stopping criterion: difference between j+1th and jth objective function values, t_max not exceeding 1000
t_max=500;
Obj_Iter_Values = zeros(t_max,1);  % Preallocate memory to store objective function values at each iteration
Vk_old = Vk;
muk_old = muk;

j = 1 ;        % Outer loop iteration count, not exceeding 10000
iner_iter = 0; % Inner loop iteration count, not exceeding 10
while j<t_max
    total_residule = calcu_residule(D_ik,U_ik,K,m);
    Obj_Iter_Values(j+1) = total_residule;

    % check for convergence 
    if abs(Obj_Iter_Values(j+1) - Obj_Iter_Values(j)) < (threshold)
        disp(['The absolute difference of total residual between iterations before and after ', num2str(j), ' is less than 1e-5']);
        break;
    end
    %   if code runs more than 10s, stop whether or not convergent
%     if toc(t) > 10
%         break;
%     end 
    j = j + 1;
    while iner_iter<5
        [Vk,muk] = update_V_b(data,U_ik,Vk_old,muk_old,K,D,m,N,alpha,lambda,weight);
       % Update Vk_old
        Vk_old = Vk;
        muk_old = muk;
        if norm(Vk-Vk_old)<1e-3
%             disp('Inner loop stopped: Vk difference less than 1e-3!')
            break
        end
        iner_iter = iner_iter + 1;
    end
    D_ik = error_ik(Vk,muk,data,N,K,alpha,lambda,weight);
    U_ik = fuzzy_u_ik(D_ik,N,K,m);
end

Obj_Iter_Values = Obj_Iter_Values(Obj_Iter_Values~=0);
% Get label information
[~,L] = max(U_ik,[],2);
% Output clustering results
% cluster_output.Outlier_hist = Outlier_hist;
% 
cluster_output.U = U_ik;
cluster_output.L = L;
cluster_output.rel = Obj_Iter_Values;


    function [Vk,muk] = inial_Vk_muk(data,U_kpc,K,D,weight)
        % Initialize Vk, muk using KPC method
        Vk = zeros(K,D);
        muk = zeros(K,D);
        for c=1:K
            % Data points belonging to the k-th plane cluster
            dk = data(U_kpc(:,c)~=0,:);
            wt_c = weight(U_kpc(:,c)~=0,:);
            aa = dk.*repmat(wt_c,1,D);
            dk_s = sum(aa,1);
            nk = sum(wt_c,1);
            % Construct matrix for eigenvector computation
            Wk = dk'*diag(wt_c)*dk - (dk_s'*dk_s)/nk;
            [v,S] = eig(Wk);
            [~,ix] = min(diag(S));
            % Normal vector of plane cluster
            Vk(c,:) = v(:,ix)';
            % Center of plane cluster
            muk(c,:) = dk_s/nk;
        end
        
    end
%% Compute fuzzy membership matrix U_ik
    function U_ik = fuzzy_u_ik(e_ik,N,K,m)
        % Input: e_ik, squared error from i-th sample to k-th plane
        % U_ik: fuzzy membership matrix U_ik, N*K
        U_ik = zeros(N,K);
        
        m_1 = 1/(m - 1);
        % Count number of zero elements per row in e_ki, e_ki_row_0
        e_ki_row_0 = sum(e_ik==0, 2);
        not_0_id = find(e_ki_row_0~=0);
        
        if ~isempty(not_0_id)
            Ii_ = zeros(N,1);
            Ii_(not_0_id,:) = 1./e_ki_row_0(not_0_id);
            U_ik = (e_ik==0).*repmat(Ii_,1,K);
        end
        % Indices of rows with no zero elements in e_ki
        id_0 = find(e_ki_row_0==0);
        n0 = length(id_0);
        for i = 1:n0
            ei = e_ik(id_0(i),:);
            for kk = 1:K
                eci_eli = (ei(kk)./ei).^m_1;
                U_ik(id_0(i),kk) = 1/sum(eci_eli);
            end
        end
        
    end
%% Compute e_ik matrix, squared error from i-th sample to cluster k
    function D_ik = error_ik(Vk,muk,data,N,K,alpha,lambda,weight)
        % Vk: K hyperplane coefficients, K*D
        % b: K*1 
        % e_ki: squared error from i-th sample to cluster k, N*K
        % N*K
        D_ik = zeros(N,K);
        for c=1:K
            vc = Vk(c,:);
            muk_c = muk(c,:);
            data_ = data - repmat(muk_c,N,1);
            % N*1
            xv = data_*vc';
            dis1 = abs(xv)*(1-alpha);
            dis2 = alpha*(xv.^2);

            dis3 = lambda*sum(data_.^2,2);
            D_ik(:,c) = dis1+ dis2 + dis3;
            D_ik(:,c) = D_ik(:,c).*weight;
        end
        
    end

    function [Vk,muk] = update_V_b(data,Uik,Vk_ini,muk_ini,K,D,m,N,alpha,lambda,weight)
        % Update hyperplane normal vectors V and K cluster centers muk
        pU = Uik.^m;
        Vk = zeros(K,D);
        muk = zeros(K,D);
        s_ik = zeros(N,K);
        g_ik = zeros(N,K);
        % One update iteration
        for c=1:K
            vc = Vk_ini(c,:);
            muk_c = muk_ini(c,:);
            data_ = data - repmat(muk_c,N,1);
            % N*1
            xv = data_*vc';
            dis1 = max(abs(xv),1e-3);
            s_ik(:,c) = 1./dis1;
            g_ik(:,c) = alpha + s_ik(:,c).*(1-alpha);
            % Construct matrix Wk to solve for eigenvector Vk_c
            temp = pU(:,c).*g_ik(:,c).*weight;
            Wk = data_'*diag(temp)*data_;
            [v,S] = eig(Wk);
            [~,ix] = min(diag(S));
            Vk(c,:) = v(:,ix)';
            % Update muk below
            vkvk = Vk(c,:)'*Vk(c,:);
            lpU = lambda*pU(:,c);
            Bk = sum(temp)*vkvk + sum(lpU)*eye(D);
            yk = sum(diag(temp)*data*vkvk) + sum(diag(lpU)*data);
            % Center of each plane cluster
            muk(c,:) = (Bk\(yk'))';
        end
    end

    function total_residule = calcu_residule(e_ik,U_ik,K,m)
        % Compute total residual sum of squares
        resid = zeros(K,1);
        for jj = 1:K
            pW_k = U_ik(:,jj).^m;
            %%%%%%%%%%%%%%%%%%%%%%%%%%% Can try computing pGamma_k without weighting first to see results
            gap = e_ik(:,jj);
            resid(jj) = sum(pW_k.*gap);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        total_residule = sum(resid);  % Compute sum of residuals from all samples to all planes
    end

end