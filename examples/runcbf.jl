# usage: julia runcbf.jl YOUR-CBF-FILE.cbf.gz

using Pajarito, ConicBenchmarkUtilities

mip_solver_drives = false

using CPLEX
mip_solver = CplexSolver(
    CPX_PARAM_SCRIND=(mip_solver_drives ? 1 : 0),
    CPX_PARAM_EPGAP=(mip_solver_drives ? 1e-5 : 0.),
    CPX_PARAM_EPINT=1e-8,
    CPX_PARAM_EPRHS=1e-7,
)

# using Mosek
# cont_solver = MosekSolver(LOG=0)

using SCS
cont_solver = SCSSolver(eps=1e-5, max_iters=100000, verbose=0)

# using ECOS
# cont_solver = ECOSSolver(verbose=false)

solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    mip_solver=mip_solver,
    cont_solver=cont_solver,
    log_level=3,
)

dat = readcbfdata(ARGS[1])
(c, A, b, con_cones, var_cones, vartypes, sense, objoffset) = cbftompb(dat)
m = MathProgBase.ConicModel(solver)
MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
MathProgBase.setvartype!(m, vartypes)
MathProgBase.optimize!(m)

@show MathProgBase.getsolvetime(m)
@show MathProgBase.getsolution(m)
@show MathProgBase.getobjval(m)
