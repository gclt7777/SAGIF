function SAHVES(Global)
% <algorithm> <M>
%SAHVES Surrogate-assisted HV environmental selection for MaOPs.
%
%   This implementation realises the surrogate-assisted hypervolume-based
%   environmental selection (SA-HVES). Hypervolume contributions are
%   estimated through a surrogate model trained on a small set of exact
%   evaluations. Each generation, a small subset of candidate solutions is
%   selected for expensive hypervolume contribution estimation via Monte
%   Carlo sampling, while the surrogate provides rapid approximations for
%   the remaining individuals. The final survivors are chosen using the
%   mixed exact and surrogate scores, and all newly evaluated points refresh
%   the surrogate model.
%
% <algorithm> <M>
%
%------------------------------- Reference --------------------------------
% L. Cheng, X. Zhang, Y. Tian, R. Cheng, and Y. Jin, Surrogate-assisted
% evolutionary algorithm for fast hypervolume-based many-objective
% optimization, IEEE Transactions on Evolutionary Computation, 2020,
% 24(6): 1146-1160.
%--------------------------------------------------------------------------

    %% Parameter setting
    [initFrac,budgetFrac,filterEps,sampleFactor,sigmaScale] = ...
        Global.ParameterSet(1.0,0.25,0.05,50,0.5);

    initFrac     = min(max(initFrac,0),1);
    budgetFrac   = min(max(budgetFrac,0),1);
    filterEps    = max(filterEps,0);
    sampleFactor = max(1,sampleFactor);
    sigmaScale   = max(sigmaScale,1e-3);

    %% Initialization
    Population = Global.Initialization();
    popSize    = length(Population);
    if popSize == 0
        return;
    end

    data.maxSize = max(200,5*Global.N);
    data.objs    = zeros(0,Global.M);
    data.hvc     = zeros(0,1);

    PopObj   = PopulationObjMatrix(Population);
    refPoint = ComputeReferencePoint(PopObj);

    numInit    = min(popSize,max(1,round(initFrac*popSize)));
    initIdx    = randperm(popSize,numInit);
    sampleCnt  = DetermineSampleCount(sampleFactor,numInit,Global.M);
    initHVC    = MonteCarloHVCSubset(PopObj,refPoint,initIdx,sampleCnt);
    data       = AppendTrainingData(data,PopObj(initIdx,:),initHVC);
    surrogate  = TrainRBFModel(data,sigmaScale);

    PopCons     = cat(1,Population.cons);
    [FrontNo,~] = NDSort(PopObj,PopCons,Global.N);
    CrowdDis    = CrowdingDistance(PopObj,FrontNo);

    %% Optimization loop
    while Global.NotTermination(Population)
        if length(Population) > 1
            MatingPool = TournamentSelection(2,2*Global.N,FrontNo,-CrowdDis);
            Offspring  = GA(Population(MatingPool));
        else
            Offspring = GA(repmat(Population,1,2));
        end

        Combined      = [Population,Offspring];
        CombinedObjs  = PopulationObjMatrix(Combined);
        CombinedCons  = cat(1,Combined.cons);
        [FrontNoCombined,MaxFront] = NDSort(CombinedObjs,CombinedCons,Global.N);
        Next       = FrontNoCombined < MaxFront;
        Population = Combined(Next);

        K = Global.N - sum(Next);
        unionObjs = CombinedObjs;
        refPoint  = ComputeReferencePoint(unionObjs);
        if K > 0
            lastIdx   = find(FrontNoCombined == MaxFront);
            lastFront = Combined(lastIdx);
            lastObjs  = CombinedObjs(lastIdx,:);
            [selectedLast,data,surrogate] = SurrogateHVSelection(lastFront,K,data,surrogate,...
                budgetFrac,filterEps,sampleFactor,sigmaScale,refPoint,Global.N,lastObjs);
            Population = [Population,selectedLast];
        end

        PopObj   = PopulationObjMatrix(Population);
        PopCons  = cat(1,Population.cons);
        [FrontNo,~] = NDSort(PopObj,PopCons,Global.N);
        CrowdDis    = CrowdingDistance(PopObj,FrontNo);
    end
end

function [Selected,data,surrogate] = SurrogateHVSelection(Front,K,data,surrogate,...
    budgetFrac,filterEps,sampleFactor,sigmaScale,refPoint,popSize,FrontObjs)
%Surrogate-assisted survivor selection on the last front.

    if nargin < 11 || isempty(FrontObjs)
        FrontObjs = PopulationObjMatrix(Front);
    end

    frontSize = length(Front);
    if frontSize <= K
        selectedIdx = 1:frontSize;
        sampleCnt   = DetermineSampleCount(sampleFactor,frontSize,size(FrontObjs,2));
        actualVals  = MonteCarloHVCSubset(FrontObjs,refPoint,selectedIdx,sampleCnt);
        data        = AppendTrainingData(data,FrontObjs(selectedIdx,:),actualVals);
        surrogate   = TrainRBFModel(data,sigmaScale);
        Selected    = Front(selectedIdx);
        return;
    end

    preds = PredictSurrogate(surrogate,FrontObjs);

    if isempty(data.objs)
        distances = inf(frontSize,1);
    else
        distances = MinimumL1Distance(FrontObjs,data.objs);
    end

    [~,order] = sort(preds,'descend');
    filtered   = order(distances(order) > filterEps);
    if isempty(filtered)
        filtered = order;
    end

    budget = max(1,min(frontSize,round(budgetFrac*popSize)));
    evalIdx = filtered(1:min(budget,length(filtered)));

    hvEstimate   = preds;
    actualIdx    = [];
    actualValues = [];
    if ~isempty(evalIdx)
        sampleCnt   = DetermineSampleCount(sampleFactor,numel(evalIdx),size(FrontObjs,2));
        evaluated   = MonteCarloHVCSubset(FrontObjs,refPoint,evalIdx,sampleCnt);
        hvEstimate(evalIdx) = evaluated;
        actualIdx          = evalIdx(:);
        actualValues       = evaluated(:);
    end

    [~,rank]    = sort(hvEstimate,'descend');
    selectedIdx = rank(1:K);

    missing = setdiff(selectedIdx,actualIdx);
    if ~isempty(missing)
        sampleCnt   = DetermineSampleCount(sampleFactor,numel(missing),size(FrontObjs,2));
        missingVals = MonteCarloHVCSubset(FrontObjs,refPoint,missing,sampleCnt);
        hvEstimate(missing) = missingVals;
        actualIdx           = [actualIdx; missing(:)];
        actualValues        = [actualValues; missingVals(:)];
    end

    Selected = Front(selectedIdx);

    if ~isempty(actualIdx)
        data      = AppendTrainingData(data,FrontObjs(actualIdx,:),actualValues);
        surrogate = TrainRBFModel(data,sigmaScale);
    end
end

function objs = PopulationObjMatrix(Pop)
%Collect the objective matrix from an array of individuals.

    if isempty(Pop)
        objs = zeros(0,0);
    else
        objs = cat(1,Pop.objs);
    end
end

function refPoint = ComputeReferencePoint(PopObj)
%Construct a slightly expanded nadir-based reference point.

    worst = max(PopObj,[],1);
    best  = min(PopObj,[],1);
    span  = max(worst-best,1e-6);
    refPoint = worst + 0.1*span;
end

function sampleCnt = DetermineSampleCount(sampleFactor,targetCount,numObj)
%Determine the Monte Carlo sample budget for HV estimation.

    base       = max(1,targetCount);
    sampleCnt  = round(sampleFactor*base*max(1,sqrt(numObj)));
    sampleCnt  = max(200,sampleCnt);
end

function data = AppendTrainingData(data,newObjs,newHVC)
%Append new training samples while enforcing the buffer size.

    if isempty(newObjs)
        return;
    end
    data.objs = [data.objs; newObjs];
    data.hvc  = [data.hvc; newHVC(:)];
    excess = size(data.objs,1) - data.maxSize;
    if excess > 0
        data.objs(1:excess,:) = [];
        data.hvc(1:excess)    = [];
    end
end

function surrogate = TrainRBFModel(data,sigmaScale)
%Train a radial basis surrogate to approximate HV contributions.

    surrogate = struct('trained',false,'centers',[],'alpha',[],'sigma',1,...
        'bias',0,'min',[],'range',[]);

    if isempty(data.objs) || size(data.objs,1) < 3
        return;
    end

    Y = data.objs;
    t = data.hvc(:);

    minY  = min(Y,[],1);
    maxY  = max(Y,[],1);
    range = max(maxY-minY,1e-6);
    Ynorm = (Y-minY)./range;

    distMatrix = pdist2(Ynorm,Ynorm);
    distMatrix(1:size(distMatrix,1)+1:end) = 0;
    upperTri = distMatrix(triu(true(size(distMatrix)),1));
    upperTri = upperTri(upperTri>0);
    if isempty(upperTri)
        medianDist = 1;
    else
        medianDist = median(upperTri);
        if isnan(medianDist) || medianDist <= 0
            medianDist = mean(upperTri);
            if isnan(medianDist) || medianDist <= 0
                medianDist = 1;
            end
        end
    end

    sigma = sigmaScale*medianDist;
    if sigma <= 0
        sigma = 1;
    end

    K = exp(-(distMatrix.^2)/(2*sigma^2));
    lambda = 1e-6;
    bias   = mean(t);
    alpha  = (K + lambda*eye(size(K))) \ (t-bias);

    surrogate.trained = true;
    surrogate.centers = Ynorm;
    surrogate.alpha   = alpha;
    surrogate.sigma   = sigma;
    surrogate.bias    = bias;
    surrogate.min     = minY;
    surrogate.range   = range;
end

function scores = PredictSurrogate(surrogate,objs)
%Predict HV contribution scores using the surrogate model.

    if isempty(surrogate) || ~surrogate.trained
        scores = zeros(size(objs,1),1);
        return;
    end

    normObjs = (objs-surrogate.min)./surrogate.range;
    normObjs(~isfinite(normObjs)) = 0;
    normObjs = max(min(normObjs,5),-5);

    distMatrix = pdist2(normObjs,surrogate.centers);
    K = exp(-(distMatrix.^2)/(2*surrogate.sigma^2));
    scores = surrogate.bias + K*surrogate.alpha;
end

function distances = MinimumL1Distance(points,reference)
%Compute the minimum L1 distance from each point to the reference set.

    if isempty(reference)
        distances = inf(size(points,1),1);
        return;
    end
    distances = min(pdist2(points,reference,'cityblock'),[],2);
end

function values = MonteCarloHVCSubset(points,refPoint,targetIdx,sampleCnt)
%Approximate HV contributions for a subset of individuals via sampling.

    targetIdx = unique(targetIdx(:)');
    values    = zeros(length(targetIdx),1);
    if isempty(targetIdx) || isempty(points)
        return;
    end

    [N,M] = size(points);
    refVec = refPoint(:)';
    clipped = min(points,repmat(refVec,N,1)-1e-9);

    lower = min(clipped,[],1);
    lower = min(lower,refVec-1e-9);
    range = refVec - lower;
    range(range <= 1e-9) = 1e-9;

    Samples = lower + rand(sampleCnt,M).*range;
    mask    = bsxfun(@le,clipped,permute(Samples,[3 2 1]));
    domMask = reshape(all(mask,2),[N,sampleCnt]);
    dominators = sum(domMask,1);
    uniqueCols = dominators == 1;

    if any(uniqueCols)
        targetMask = domMask(targetIdx,uniqueCols);
        counts     = sum(targetMask,2);
        volume     = prod(range);
        values     = counts/sampleCnt * volume;
    end

    if all(values == 0)
        values = values + (1e-12*(1:length(values))');
    end
end