function [results, num_solved] = runSingleExperiment(problem, N, Nts, num_iter, tol)
    fprintf('\tSolving %s\n', problem.Name);
    [Xtr, Ytr, Xts, Yts] = problem.loaddata();
    fprintf('\t\tOriginal training size n = %d, d = %d\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, N, Nts);
    fprintf('\t\tSubsampled training size n = %d, d = %d\n\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Xts] = standarize(Xtr, Xts);

    A = kernelmatrix(Xtr, Xtr, problem.Kernel, problem.Bandwidth);
    Ats = kernelmatrix(Xts, Xtr, problem.Kernel, problem.Bandwidth);
    test_accuracy = @(beta) mean(2 * abs(Ats*beta - Yts) ./ (abs(Ats*beta) + abs(Yts)));
    relres = @(beta) norm(A*beta + problem.Mu*beta - Ytr) / norm(Ytr);
    summary = @(beta) [relres(beta) test_accuracy(beta)];

    results = struct();
    % Run KRR with different methods and store results
    [~, results.rpc] = krr(A, problem.Mu, Ytr, problem.ApproximationRank, [], summary, 'rpcnys', num_iter, tol, tol);
    [~, results.greedy] = krr(A, problem.Mu, Ytr, problem.ApproximationRank, [], summary, 'greedynys', num_iter, tol, tol);
    [~, results.uniform] = krr(A, problem.Mu, Ytr, problem.ApproximationRank, [], summary, 'uninys', num_iter, tol, tol);
    [~, results.nopre] = krr(A, problem.Mu, Ytr, problem.ApproximationRank, [], summary, '', num_iter, tol, tol);

    num_solved = struct();
    % Determine if accuracy within 50 iterations is achieved
    num_solved.rpc = min(find(results.rpc(:,1) <= 1e-3, 50)) <= 50;
    num_solved.greedy = min(find(results.greedy(:,1) <= 1e-3, 50)) <= 50;
    num_solved.uniform = min(find(results.uniform(:,1) <= 1e-3, 50)) <= 50;
    num_solved.nopre = min(find(results.nopre(:,1) <= 1e-3, 50)) <= 50;
end