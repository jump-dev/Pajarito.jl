# These problems with duality gaps can cause Pajarito to fail, because we cannot detect duality gaps

using MathProgBase, Pajarito
log_level = 3

# using ECOS
# cont_solver = ECOSSolver(verbose=false)

using Mosek
cont_solver = MosekSolver(LOG=0)

# using Cbc
# mip_solver = CbcSolver()
# mip_solver_drives = false

using CPLEX
mip_solver = CplexSolver()
mip_solver_drives = false


solver = PajaritoSolver(
	mip_solver_drives=mip_solver_drives,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
	log_level=log_level,
	soc_disagg=false,
	soc_abslift=false,
	init_soc_one=false,
	init_soc_inf=false,
	prim_cuts_assist=true
)


# Infinite duality gap
# Example of polyhedral OA failure due to infinite duality gap from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
# min  z
# st   x == 0
#     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
#      x in {0,1}

m = MathProgBase.ConicModel(solver)
MathProgBase.loadproblem!(m,
[ 0.0, 0.0, 1.0],
[ -1.0  0.0  0.0;
-1.0  0.0  0.0;
0.0 -1.0  0.0;
0.0  0.0 -1.0],
[ 0.0, 0.0, 0.0, 0.0],
Any[(:Zero,1:1),(:SOCRotated,2:4)],
Any[(:Free,[1,2,3])])
MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])

MathProgBase.optimize!(m)

@show MathProgBase.status(m)
@show MathProgBase.getobjval(m)
@show MathProgBase.getsolution(m)


# Finite duality gap
# Example of polyhedral OA failure due to finite duality gap, modified from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
# min  z
# st   x == 0
#     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
#      z >= -10
#      x in {0,1}

m = MathProgBase.ConicModel(solver)
MathProgBase.loadproblem!(m,
[ 0.0, 0.0, 1.0],
[ -1.0  0.0  0.0;
 -1.0  0.0  0.0;
  0.0 -1.0  0.0;
  0.0  0.0 -1.0;
  0.0  0.0 -1.0],
[ 0.0, 0.0, 0.0, 0.0, 10.0],
Any[(:Zero,1:1),(:SOCRotated,2:4),(:NonNeg,5:5)],
Any[(:Free,[1,2,3])])
MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])

MathProgBase.optimize!(m)

@show MathProgBase.status(m)
@show MathProgBase.getobjval(m)
@show MathProgBase.getsolution(m)
