# This problem should cause Pajarito to fail, because we cannot detect duality gaps
# Example of polyhedral OA failure due to infinite duality gap from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
# min  z
# st   x == 0
#     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
#      x in {0,1}

using MathProgBase, Pajarito
log_level = 2

using ECOS
cont_solver = ECOSSolver(verbose=false)

# using Cbc
# mip_solver = CbcSolver()
# mip_solver_drives = false

using CPLEX
mip_solver = CplexSolver()
mip_solver_drives = true


solver = PajaritoSolver(
	mip_solver_drives=mip_solver_drives,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
	log_level=log_level
)


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
