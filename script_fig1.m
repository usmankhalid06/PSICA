clear;
close all; 
clc;


nV = 70; 
nSRCS = 7; %number of sources
nIter = 30; %algorithm iterations
N = 300; %No. of time points
tstd  = sqrt(0.9); %0.6 is the varaince
sstd  = sqrt(0.01);
Dp = dctbases(N,N); %dct basis dictionary


load TC 
load SM
rng('default');
rng('shuffle') % random number generator
Y= (TC+tstd*randn(N,nSRCS))*(SM+sstd*randn(nSRCS,nV^2));
Y= Y-repmat(mean(Y),size(Y,1),1);
K =9;


%% ICA
tic
[G,~,~] = svds(Y,K);
Ss = G'*Y;
[SSs,~,~] = fastica(Ss, 'numOfIC', K,'approach','symm', 'g', 'tanh','verbose', 'off');
X{1} = SSs;
D{1} = Y*SSs';
toc

%% PMD
tic
K1 = 24; K2 = 16;
lambda =0.5; gamma=lambda*ones(1,K1);
tmpX=GPower(Y,gamma,K1,'l1',0);
Y2 = Y*tmpX;
Y2 = Y2*diag(1./sqrt(sum(Y2.*Y2)));
lambda2 = 0.5;
[Wx1,~,~] = pmd_rankK(Y',Y2',K2,lambda2);
X{2} = Wx1';
D{2} = Y*X{2}';
toc

%% SICA-EBM
tic
[G,~,~] = svds(Y,K);
Ss = G'*Y;
[WW,~,~,~,~] = ICA_EBM_Sparse(Ss,5,10^5); %0.7
D{3} = Y*(real(WW)*Ss)';
X{3} = WW*Ss;
toc

%% ssBSS
tic
params1.K = 9;
params1.P = 7; %8
params1.lam1 = 6; %6
params1.zeta1 = 120;
params1.Kp = 180;
params1.nIter = nIter;
params1.alpha = 10^-8; %1e-9
[D{4},X{4},~,~]=ssBSS_pre(Y,Dp,params1,TC,SM); %_mod
toc

%% SICA-L
tic
spa = 0.01; %0.015
[D{5},X{5},U,Err]=LSICA(Y,K,spa,nIter,TC,SM);  D{5} = D{5} * diag(1./sqrt(sum(D{5}.*D{5}))); 
toc

%% PSICA
tic
TR = 1; N = 300;
Dp1 = easy_hrf_basis(N, TR, 1200, 'fast');     % Brief events
Dp2 = easy_hrf_basis(N, TR, 1150, 'medium');   % Medium tasks  
Dp3 = easy_hrf_basis(N, TR, 1250, 'slow');     % Long blocks
Dp4 = easy_hrf_basis(N, TR, 1100, 'mixed');    % Everything
Dp_HRF= [Dp1 Dp2 Dp3 Dp4];
Dp_DCT = Dp(:,1:150);
[~,Dpp] = remove_dependent_qr([Dp_HRF Dp_DCT], 1e-12);   %
Dp_SPLINES = create_spline_dictionary(N, 150);
t = (1:N)'/N; 
Dp_smooth_bases = [t, t.^2, t.^3, sin(2*pi*t), cos(2*pi*t)];
Up2 =[Dpp Dp_smooth_bases Dp_SPLINES]; %
Uq = Up2 * diag(1./sqrt(sum(Up2.*Up2)));


spa = 0.001; [D1,X1,U,Err]=LSICA(Y,K,spa,nIter,TC,SM);  D1 = D1 * diag(1./sqrt(sum(D1.*D1))); 
[inddd,~] = remove_dependent_qr(D1, 1e-12);   
result = setdiff([1:K], inddd);
D1(:,result) = []; X1(result,:)=[];
spa1 = 36; spa2 = 4; spa3 =12; %4,12
[D{6},X{6},Err]= PSICA(Y,[D1],[X1],K-length(result),spa1,spa2,spa3,Uq,120,0.1,nIter, TC,SM);  %-length(inddd)
figure; plot(Err)
toc


%%
nA=7; 
sD{1} = TC;
sX{1} = SM;
for jj =1:nA-1
 [sD{jj+1},sX{jj+1},ind]=sort_TSandSM_spatial(TC,SM,D{jj},X{jj},nSRCS);
%     [sD{jj+1},sX{jj+1},ind]=sort_TSandSM_temporal(TC,D{jj},X{jj});
for ii =1:nSRCS
 TCcorr(jj+1,ii,1) =abs(corr(TC(:,ii),D{jj}(:,ind(ii))));
 SMcorr(jj+1,ii,1) =abs(corr(SM(ii,:)',X{jj}(ind(ii),:)'));
end
end

ccTC = mean(TCcorr(:,:,1),3)
ccSM = mean(SMcorr(:,:,1),3)
TT(1,:) = mean(ccTC');
TT(2,:) = mean(ccSM');

TT

f = figure;
f.Position = [170 120 1500 850];
for j = 1:nSRCS 
    row_base = (j-1) * 14;
    for i = 1:nA 
        subplot_tight(nSRCS+1, 14, row_base + i, [0.02 0.01]);
        plot(zscore(sD{i}(:,j)), 'b', 'LineWidth', 0.25);
        axis tight;
        ylim([-3 3]);
        ax = gca;
        ax.XTick = [1 length(sD{i}(:,j))/2 length(sD{i}(:,j))];
        ax.XTickLabel = {'0', num2str(round(length(sD{i}(:,j))/2)), num2str(length(sD{i}(:,j))-1)};
        set(gca,'YTickLabel','')
        grid on;
        subplot_tight(nSRCS+1, 14, row_base + 7 + i, [0.02 0.01]); % Changed from 8 to 7
        imagesc(flipdim(reshape(abs(zscore(sX{i}(j,:))), nV, nV), 1));
        colormap('hot');
        set(gca,'XTickLabel','')
        set(gca,'YTickLabel','')
    end
end
bottom_row_base = nSRCS * 14; 
for i = 1:nA
    subplot_tight(nSRCS+1, 14, bottom_row_base + i, [0.04 0.01]);
    if i == 1
        text(0.5, 0.5, '(a)', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
    else
        letter = sprintf('(%s)', char('a' + i - 1));
        temp_corr = sprintf('γ_T=%.2f', TT(1,i));
        text(0.5, 0.7, letter, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
        text(0.5, 0.3, temp_corr, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
    end
    axis off;
    subplot_tight(nSRCS+1, 14, bottom_row_base + 7 + i, [0.04 0.01]); 
    if i == 1
        text(0.5, 0.5, '(a)', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
    else
        letter = sprintf('(%s)', char('a' + i - 1));
        spatial_corr = sprintf('γ_S=%.2f', TT(2,i));
        text(0.5, 0.7, letter, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
        text(0.5, 0.3, spatial_corr, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
    end
    axis off;
end
exportgraphics(gcf,'khali1.png','Resolution',300)

