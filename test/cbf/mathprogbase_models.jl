# MathProgBase test models, converted to CBF for conic model tests

using ConicBenchmarkUtilities


name = "soc_equality"
# max  y + z
# st   x == 1
#     (x,y,z) in SOC
#      x in {0,1}
c = [0.0, -1.0, -1.0]
A = [1.0  0.0  0.0;
    -1.0  0.0  0.0;
     0.0 -1.0  0.0;
     0.0  0.0 -1.0]
b = [1.0, 0.0, 0.0, 0.0]
con_cones = [(:Zero,1:1), (:SOC,2:4)]
var_cones = [(:Free,[1,2,3])]
var_types = [:Int,:Cont,:Cont]
dat = ConicBenchmarkUtilities.mpbtocbf(name, c, A, b, con_cones, var_cones, var_types)
ConicBenchmarkUtilities.writecbfdata(joinpath(pwd(), "$name.cbf"), dat)


name = "soc_zero"
# Same as soc_equality, with some zero variable cones
c = [0.0, 0.0, -1.0, 1.0, -1.0]
A = [1.0  1.0  0.0  0.0  0.0;
    -1.0  0.0  0.0 -0.5  0.0;
     0.0  2.0 -1.0  0.0  0.0;
     0.0  0.0  0.0 0.5  -1.0]
b = [1.0, 0.0, 0.0, 0.0]
con_cones = [(:Zero,1:1), (:SOC,2:4)]
var_cones = [(:Free,[1,3,5]), (:Zero,[2,4])]
var_types = [:Int, :Int, :Cont, :Cont, :Cont]
dat = ConicBenchmarkUtilities.mpbtocbf(name, c, A, b, con_cones, var_cones, var_types)
ConicBenchmarkUtilities.writecbfdata(joinpath(pwd(), "$name.cbf"), dat)


name = "socrot_optimal"
# Rotated-SOC problem
c = [-3.0, 0.0, 0.0, 0.0]
A = zeros(4,4)
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0
A[4,1] = 1.0
A[4,4] = -1.0
b = [10.0, 1.5, 3.0, 0.0]
con_cones = [(:NonNeg,[1,2,3]), (:Zero,[4])]
var_cones = [(:SOCRotated,[2,3,1]), (:Free,[4])]
var_types = [:Cont, :Cont, :Cont, :Int]
dat = ConicBenchmarkUtilities.mpbtocbf(name, c, A, b, con_cones, var_cones, var_types)
ConicBenchmarkUtilities.writecbfdata(joinpath(pwd(), "$name.cbf"), dat)


name = "socrot_infeasible"
# Rotated-SOC problem
c = [-3.0, 0.0, 0.0, 0.0]
A = zeros(4,4)
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0
A[4,1] = 1.0
A[4,4] = -1.0
b = [10.0, -1.5, 3.0, 0.0]
con_cones = [(:NonNeg,[1,2,3]), (:Zero,[4])]
var_cones = [(:SOCRotated,[2,3,1]), (:Free,[4])]
var_types = [:Cont, :Cont, :Cont, :Int]
dat = ConicBenchmarkUtilities.mpbtocbf(name, c, A, b, con_cones, var_cones, var_types)
ConicBenchmarkUtilities.writecbfdata(joinpath(pwd(), "$name.cbf"), dat)


name = "exp_sepcut"
# Exp problem designed to test the special separation cuts for s = 0
# min -t
# s.t. (t,s,r)     ∈ Exp
#      1/2 - t - s ≥ 0
# s Binary
c = [-1.0, 0.0, 0.0]
A = [1.0 1.0 0.0]
b = [0.5]
con_cones = [(:NonNeg,1:1)]
var_cones = [(:ExpPrimal,1:3)]
var_types = [:Cont,:Bin,:Cont]
dat = ConicBenchmarkUtilities.mpbtocbf(name, c, A, b, con_cones, var_cones, var_types)
ConicBenchmarkUtilities.writecbfdata(joinpath(pwd(), "$name.cbf"), dat)


name = "exp_ising"
# Exp problem derived from Ising model for pairwise binary Markov random field structure estimation (work with Marc Vuffray, LANL 2016)
c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
b = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0]
I = [32, 41, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 33, 42, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 34, 43, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 35, 44, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 36, 45, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 37, 46, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 38, 47, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 39, 48, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 40, 49, 3, 31, 6, 31, 9, 31, 12, 31, 15, 31, 18, 31, 21, 31, 24, 31, 27, 31, 30, 31, 31, 32, 41, 50, 51, 33, 42, 50, 34, 43, 50, 35, 44, 50, 36, 45, 50, 37, 46, 50, 38, 47, 50, 39, 48, 50, 40, 49, 50]
J = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29]
V = [-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, -1.0, -0.1, 1.0, -0.3, -0.3, 1.0, -1.0, -0.3, -0.3, 1.0, -0.3, -0.3, 1.0, -0.3, -0.3, 1.0, -0.3, -0.3, 1.0, -0.3, -0.3, 1.0, -0.3, -0.3, 1.0, -0.3, -0.3, 1.0, -0.3, -0.3, 1.0]
A = sparse(I, J, V, length(b), length(c))
con_cones = [(:ExpPrimal,1:3), (:ExpPrimal,4:6), (:ExpPrimal,7:9), (:ExpPrimal,10:12), (:ExpPrimal,13:15), (:ExpPrimal,16:18), (:ExpPrimal,19:21), (:ExpPrimal,22:24), (:ExpPrimal,25:27), (:ExpPrimal,28:30), (:Zero,31:31), (:NonNeg,32:40), (:NonNeg,41:49), (:NonNeg,50:50), (:Zero,51:51)]
var_cones = [(:Free,1:29)]
var_types = [:Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Cont, :Bin, :Bin, :Bin, :Bin, :Bin, :Bin, :Bin, :Bin, :Bin]
dat = ConicBenchmarkUtilities.mpbtocbf(name, c, A, b, con_cones, var_cones, var_types)
ConicBenchmarkUtilities.writecbfdata(joinpath(pwd(), "$name.cbf"), dat)
