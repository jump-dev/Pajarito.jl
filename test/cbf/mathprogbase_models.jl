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
con_cones = Any[(:Zero,1:1),(:SOC,2:4)]
var_cones = Any[(:Free,[1,2,3])]
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
con_cones = Any[(:Zero,1:1), (:SOC,2:4)]
var_cones = Any[(:Free,[1,3,5]), (:Zero,[2,4])]
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
con_cones = Any[(:NonNeg,[1,2,3]), (:Zero,[4])]
var_cones = Any[(:SOCRotated,[2,3,1]), (:Free,[4])]
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
con_cones = Any[(:NonNeg,[1,2,3]), (:Zero,[4])]
var_cones = Any[(:SOCRotated,[2,3,1]), (:Free,[4])]
var_types = [:Cont, :Cont, :Cont, :Int]
dat = ConicBenchmarkUtilities.mpbtocbf(name, c, A, b, con_cones, var_cones, var_types)
ConicBenchmarkUtilities.writecbfdata(joinpath(pwd(), "$name.cbf"), dat)
