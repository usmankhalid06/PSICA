function [U,Z,Err,C]= PSICA(Y,Uq,Zq,K,lambda1,lambda2,lambda3,Up,zeta3,tau1,nIter,TC,SM)
    Psi = eye(K,K);
    Phi = zeros(K,K);
    Lt = size(Up,2);
    U = Uq*Psi;
    Z = Phi*Zq;
    sizeU = numel(U);
    sizeZ = numel(Z');
    U_tol = 1e-9;
    
    % Only add Gram matrix for the one optimization you want
    Gram_Up = Up'*Up;
    
    fprintf('Iteration: ');
    for iter=1:nIter
        fprintf('\b\b\b\b\b%5i',iter);
        Do = U;
        F1 = U'*U; E1 = U'*Y;
        iiter = 0;
        Zpp = Z;
        while (iiter < nIter)
            iiter = iiter + 1;
            for i =1:K
                xk = 1.0/F1(i,i) * (E1(i,:) - F1(i,:)*Z) + Z(i,:);
                thr1 = lambda1./abs(xk);
                Z(i,:) = sign(xk).*max(0, bsxfun(@minus,abs(xk),thr1/2));
            end
            %% check stop condition
            if (norm(Z - Zpp, 'fro')/sizeZ < U_tol)
                break;
            end
            Zpp = Z;
        end
        Phi = Z*pinv(Zq);
        Z = Phi*Zq;

        for j=1:K
            Z(j,:) = firm_thresholding_nonadaptive(Z(j,:), lambda2/2, lambda3/2);  
        end

        F2 = Z*Z'; E2 = Y*Z';
        iiter = 0;
        Upp = U;
        while (iiter < nIter)
            iiter = iiter + 1;
            for j = 1: K
                if(F2(j,j) ~= 0)
                    tmp3 = 1.0/F2(j,j) * (E2(:,j) - U*F2(:, j)) + U(:,j);
                    Psi(:,j)= pinv(Uq)*tmp3;
                    Psi(:,j) = Psi(:,j)./(max( norm(Uq*Psi(:,j),2),1));
                    U(:,j) = Uq*Psi(:,j);
                end
            end
            %% check stop condition
            if (norm(U - Upp, 'fro')/sizeU < U_tol)
                break;
            end
            Upp = U;
        end
        A = zeros(Lt, K);
        all_correlations = abs(Up' * U);
        for k = 1:K
            correlations = all_correlations(:, k);
            maxCorr = max(correlations);
            threshold = tau1 * maxCorr;
            [~, bb] = sort(correlations, 'descend');
            ind = bb(1:zeta3);
            keepInd = ind(correlations(ind) >= threshold);
            A(:, k) = 0; % Zero out all elements
            if ~isempty(keepInd)
                G_keep = Gram_Up(keepInd, keepInd);
                b_keep = Up(:, keepInd)' * U(:, k);
                A(keepInd, k) = pinv(G_keep)*b_keep; %G_keep \ b_keep; %
                A(:,k) = A(:,k)./norm(Up*A(:,k));
                U(:, k) = Up * A(:, k);
            end
        end
        Err(iter)= sqrt(trace((U-Do)'*(U-Do)))/sqrt(trace(Do'*Do));
        K2 = size(TC,2);
        [~,~,ind]=sort_TSandSM_spatial(TC,SM,U,Z,K2);
        for ii =1:K2
            TCcorr(ii) =abs(corr(TC(:,ii),U(:,ind(ii))));
            SMcorr(ii) =abs(corr(SM(ii,:)',Z(ind(ii),:)'));
        end
        cTC = sum(TCcorr');
        cSM = sum(SMcorr');
        C(iter) =cTC+cSM;
    end
    ind = find(sum(abs(Z'))==0);
    Z(ind,:) = sign(xk).*max(0, bsxfun(@minus,abs(U(:,ind)'*Y),lambda1/5));

end

% function [U,Z,Err,Psi,Phi]= PSICA(Y,Uq,Zq,K,lambda1,lambda2,lambda3,Up,zeta3,tau1,nIter,TC,SM)
% 
%     % Relaxed tolerance
%     U_tol = 1e-6;  % Less strict tolerance
%     Z_tol = 1e-6;
%     
%     % Add damping factors
%     alpha_u = 0.3;  % Damping for U updates
%     alpha_z = 0.3;  % Damping for Z updates
%     
%     % Limit inner iterations
%     max_inner_iter = 10;  % Much fewer inner iterations
%     
%     % Initialize
%     Psi = eye(K,K);
%     Phi = zeros(K,K);
%     Lt = size(Up,2);
%     U = Uq*Psi;
%     Z = Phi*Zq;
%     
%     Gram_Up = Up'*Up;
%     fprintf('Iteration: ');
% 
%     for iter=1:nIter
%         fprintf('\b\b\b\b\b%5i',iter);
%         Do = U;
%         
%         % Z update with limited inner iterations
%         F1 = U'*U+ 1e-6*eye(K); E1 = U'*Y;
%         Z_old = Z;
%         
%         for inner_z = 1:max_inner_iter  % Limited iterations
%             Z_prev = Z;
%             for i = 1:K
%                 xk = 1.0/F1(i,i) * (E1(i,:) - F1(i,:)*Z) + Z(i,:);
%                 thr1 = lambda1./abs(xk);
%                 Z_new = sign(xk).*max(0, bsxfun(@minus,abs(xk),thr1/2));
%                 Z(i,:) = alpha_z * Z_new + (1-alpha_z) * Z(i,:);  % Damping
%             end
%             
%             if norm(Z - Z_prev, 'fro') < Z_tol * norm(Z, 'fro')
%                 break;
%             end
%         end
%         
%         % Apply firm thresholding
%         Phi = Z/Zq;
%         Z = Phi*Zq;
%         for j=1:K
%             Z(j,:) = firm_thresholding_nonadaptive(Z(j,:), lambda2/2, lambda3/2);
%         end
%         
%         % U update with limited inner iterations
%         F2 = Z*Z'+ 1e-6*eye(K); E2 = Y*Z';
%         U_old = U;
%         
%         for inner_u = 1:max_inner_iter  % Limited iterations
%             U_prev = U;
%             for j = 1:K
%                 if F2(j,j) ~= 0
%                     tmp3 = 1.0/F2(j,j) * (E2(:,j) - U*F2(:,j)) + U(:,j);
%                     Psi_new = pinv(Uq)*tmp3;
%                     
%                     % Consistent normalization
%                     Psi_new = Psi_new ./ max(norm(Psi_new), 1e-8);
%                     U_new = Uq * Psi_new;
%                     
%                     % Damping
%                     Psi(:,j) = alpha_u * Psi_new + (1-alpha_u) * Psi(:,j);
%                     U(:,j) = Uq * Psi(:,j);
%                 end
%             end
%             
%             if norm(U - U_prev, 'fro') < U_tol * norm(U, 'fro')
%                 break;
%             end
%         end
%         
%         % Your A matrix update (this looks fine)
%         A = zeros(Lt, K);
%         all_correlations = abs(Up' * U);
%         for k = 1:K
%             correlations = all_correlations(:, k);
%             maxCorr = max(correlations);
%             threshold = tau1 * maxCorr;
%             [~, bb] = sort(correlations, 'descend');
%             ind = bb(1:zeta3);
%             keepInd = ind(correlations(ind) >= threshold);
%             
%             if ~isempty(keepInd)
%                 G_keep = Gram_Up(keepInd, keepInd);
%                 b_keep = Up(:, keepInd)' * U(:, k);
%                 A(keepInd, k) = pinv(G_keep)*b_keep;
%                 A(:,k) = A(:,k)./max(norm(Up*A(:,k)), 1e-8);  % Avoid division by zero
%                 U(:, k) = Up * A(:, k);
%             end
%         end
%         
%         Err(iter) = norm(U-Do,'fro')/max(norm(Do,'fro'), 1e-8);
%         K2 = size(TC,2);
%         [~,~,ind]=sort_TSandSM_spatial(TC,SM,U,Z,K2);
%         for ii =1:K2
%             TCcorr(ii) =abs(corr(TC(:,ii),U(:,ind(ii))));
%             SMcorr(ii) =abs(corr(SM(ii,:)',Z(ind(ii),:)'));
%         end
%         cTC = sum(TCcorr');
%         cSM = sum(SMcorr');
%         C(iter) =cTC+cSM;        
%         
%         % Early stopping
%         if iter > 1 && Err(iter) < 1e-6
%             break;
%         end
%     end
% end


function x_firm = firm_thresholding_nonadaptive(x, lambda1, lambda2)
    % Firm thresholding function
    % Inputs:
    %   x       : Input data (can be a vector or matrix)
    %   lambda1 : Lower threshold (soft thresholding threshold)
    %   lambda2 : Upper threshold (hard thresholding threshold)
    % Output:
    %   x_firm  : Output data after applying firm thresholding

    % Initialize the output array
    x_firm = zeros(size(x));
    
    % Apply firm thresholding
    for i = 1:length(x)
        if abs(x(i)) <= lambda1
            % Set to zero if below lower threshold (like hard thresholding)
            x_firm(i) = 0;
        elseif abs(x(i)) > lambda1 && abs(x(i)) <= lambda2
            % Apply partial shrinkage between the two thresholds
            x_firm(i) = sign(x(i)) * (lambda2 * (abs(x(i)) - lambda1) / (lambda2 - lambda1));
        else
            % Keep the value if above the upper threshold
            x_firm(i) = x(i);
        end
    end
end














