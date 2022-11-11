classdef ProblemParameters
   properties
      Name 
      Bandwidth
      Mu
      ApproximationRank
      Kernel
   end
   methods
       function obj = ProblemParameters(name, bandwidth, mu, approximationRank, kernel)
          obj.Name = name;
          obj.Bandwidth = bandwidth;
          obj.Mu = mu;
          obj.ApproximationRank = approximationRank;
          obj.Kernel = kernel;
       end
       function [Xtr, Ytr, Xts, Yts] = loaddata(obj)
           load("../data/preprocessed/" + obj.Name + ".mat");
           Ytr = Ytr(:);
           Yts = Yts(:);
       end
end
end
