clear;  clear memory; clc;
addpath('./Caltech101-Datasets')
addpath('./Utility')

load Caltech101-all
clear categories feanames lenSmp

% Nonlinear anchor feature embedding
viewNum = size(X,2);
load Anchor_Caltech101
fprintf('Nonlinear Anchor Embedding...\n');
for it = 1:viewNum
%     fprintf('The %d-th view Nonlinear Anchor Embeeding...\n',it);
    dist = EuDist2(X{it},Anchor{it},0);
    sigma = mean(min(dist,[],2).^0.5)*2;
    feaVec = exp(-dist/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));% Centered data 
end
clear feaVec dist sigma dist Anchor it

%------------Initializing parameters--------------
MaxIter = 7;       % 5 iterations are okay, but better results for 10
innerMax = 10;
r = 5;              % r is the power of alpha_i
L = 128;            % Hashing code length
beta = 0.003;       % Hyper-para beta
gamma = .01;        % Hyper-para gamma
lambda = 0.00001;   % Hyper-para lambda

N = size(X{1},2);
rand('seed',100);
sel_sample = X{4}(:,randsample(N, 1000),:);
[pcaW, ~] = eigs(cov(sel_sample'), L);
B = sign(pcaW'*X{4});

n_cluster = numel(unique(Y));
alpha = ones(viewNum,1) / viewNum;
U = cell(1,viewNum);

rand('seed',500);
C = B(:,randsample(N, n_cluster));
HamDist = 0.5*(L - B'*C);
[~,ind] = min(HamDist,[],2);
G = sparse(ind,1:N,1,n_cluster,N,N);
G = full(G);
CG = C*G;

XXT = cell(1,viewNum);
for view = 1:viewNum
    XXT{view} = X{view}*X{view}';
end
clear HamDist ind initInd n_randm pcaW sel_sample view
%------------End Initialization--------------

disp('----------The proposed method (multi-view)----------');
for iter = 1:MaxIter
    fprintf('The %d-th iteration...\n',iter);
    %---------Update Ui--------------
    alpha_r = alpha.^r;
    UX = zeros(L,N);
    for v = 1:viewNum
        U{v} = B*X{v}'/((1-gamma)*XXT{v}+beta*eye(size(X{v},1)));
        UX   = UX+alpha_r(v)*U{v}*X{v};
    end
    
    %---------Update B--------------
    B = sign(UX+lambda*CG);B(B==0) = -1;
    % clear UX CG
    
    %---------Update C and G--------------
    for iterInner = 1:innerMax
        % For simplicity, directly using DPLM here
        C = sign(B*G'); C(C==0) = 1;
        rho = .001; mu = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B*G' + rho*repmat(sum(C),L,1);
            C    = sign(C-1/mu*grad); C(C==0) = 1;
        end
        
        HamDist = 0.5*(L - B'*C); % Hamming distance
        [~,indx] = min(HamDist,[],2);
        G = sparse(indx,1:N,1,n_cluster,N,N);
    end
    CG = C*G;
    % clear iterIn grad HamDist indx mu rho
    
    %---------Update alpha--------------
    h = zeros(viewNum,1);
    for view = 1:viewNum
        h(view) = norm(B-U{view}*X{view},'fro')^2 -gamma*norm(U{view}*X{view},'fro')^2 + beta*norm(U{view},'fro')^2;
    end
    H = bsxfun(@power,h, 1/(1-r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,H,sum(H)); % alpha = H./sum(H);
    % clear H h
end
disp('----------Main Iteration Completed----------');

[~,pred_label] = max(G,[],1);
res_cluster = ClusteringMeasure(Y, pred_label);
fprintf('All view results: ACC = %.4f and NMI = %.4f, Purity = %.4f\n\n',res_cluster(1),res_cluster(2),res_cluster(3));
