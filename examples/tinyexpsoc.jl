
using Convex, Pajarito
log_level = 2

using ECOS
cont_solver = ECOSSolver(verbose=false)

# using Cbc
# mip_solver = CbcSolver()
# mip_solver_drives = false

using CPLEX
mip_solver = CplexSolver()
mip_solver_drives = false


x = Convex.Variable(1, :Int)
y = Convex.Variable(1)

pr = Convex.minimize(
	-3x - y,
   	x >= 1,
   	y >= 0,
   	3x + 2y <= 10,
   	x^2 <= 5,
   	exp(y) + x <= 7
)

Convex.solve!(pr, PajaritoSolver(
	mip_solver_drives=mip_solver_drives,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
	log_level=log_level,
	soc_abslift=true,
	soc_disagg=true,
	init_soc_one=false,
	init_soc_inf=false,
	init_exp=false,
))

@show pr.status
@show pr.optval
@show x
@show y
