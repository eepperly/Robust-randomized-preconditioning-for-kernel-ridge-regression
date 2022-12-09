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
           Ytr = cast(Ytr(:), 'double');
           Yts = cast(Yts(:), 'double');
           
           % LIBSVM fails to save the right size for train and test, this
           % is a hacky fix.
           if obj.Name.contains("a9a")
               Xtr = Xtr(:, 1:122);
               Xts = Xts(:, 1:122);
           end
       end
end
end
