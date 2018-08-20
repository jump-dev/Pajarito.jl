# Navigate an object through a series of obstacles, while minimizing the jerk of the trajectory.
# The domain is broken down into a polyhedral covering, and the time interval into subintervals.
# In each subinterval, we enforce the object is contained entirely within one of the polyhedral
# subregions, which enforces that we do not run into any obstacles.
#
# The trajectory is described by polynomials p_x and p_y, which describe the x and y position,
# respectively, as a function of time. This is a polynomial optimization problem, which we relax
# into a mixed-integer sum of squares problem. The problem is infeasible for polynomials of degree
# 4 or less, and feasible for degree 5.
#
# Example written and by Joey Huchette.

using JuMP, PolyJuMP, SumOfSquares, MultivariatePolynomials
using Pajarito

# Specify Pajarito solver

mip_solver_drives = true
rel_gap = 1e-4

# using Cbc
# mip_solver = CbcSolver()

using CPLEX
mip_solver = CplexSolver(
    CPX_PARAM_SCRIND=(mip_solver_drives ? 1 : 0),
    CPX_PARAM_EPINT=1e-9,
    CPX_PARAM_EPRHS=1e-9,
    CPX_PARAM_EPGAP=(mip_solver_drives ? 1e-5 : 1e-9)
)

# using Mosek
# conic_solver = MosekSolver(LOG=0)
# conic_solver = MosekSolver(MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-6,MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-6,MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-5,MSK_DPAR_INTPNT_TOL_INFEAS=1e-8,MSK_DPAR_INTPNT_CO_TOL_MU_RED=1e-6)

micp_solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    log_level=3,
    rel_gap=rel_gap,
    mip_solver=mip_solver,
    # cont_solver=conic_solver,
    solve_relax=false,
    solve_subp=false,
    prim_cuts_only=true
)

# Create model

struct Box
    xl::Float64
    xu::Float64
    yl::Float64
    yu::Float64
end

boxes = Box[
    Box(0.0,1.0,0.0,0.3),
    Box(0.8,1.7,0.1,0.3),
    Box(1.4,1.9,0.2,0.4),
    Box(1.0,1.7,0.3,0.5),
    Box(0.5,1.4,0.4,0.6),
    Box(0.0,1.0,0.5,0.7),
    Box(0.2,1.0,0.6,0.8),
    Box(0.5,1.3,0.7,0.9),
    Box(1.0,2.0,0.7,1.0)
]

N = 8 # Number of trajectory pieces
d = 2 # dimension of space
r = 5 # dimension of polynomial trajectories
M = 2 # number of horizontal segments

domain = Box(0,M,0,1)

X₀ = Dict(:x=>0, :y=>0)
X₀′ = Dict(:x=>1, :y=>0)
X₀′′ = Dict(:x=>0, :y=>0)
X₁ = Dict(:x=>2, :y=>1)

T = linspace(0, 1, N+1)

Tmin = minimum(T)
Tmax = maximum(T)

model = SOSModel(solver=micp_solver)

@polyvar(t)
Z = monomials([t], 0:r)

@variable(model, H[1:N,boxes], Bin)

Mxl, Mxu, Myl, Myu = domain.xl, domain.xu, domain.yl, domain.yu
p = Dict()
for j in 1:N
    @constraint(model, sum(H[j,box] for box in boxes) == 1)

    p[(:x,j)] = @polyvariable(model, _, Z)
    p[(:y,j)] = @polyvariable(model, _, Z)
    for box in boxes
        xl, xu, yl, yu = box.xl, box.xu, box.yl, box.yu
        @polyconstraint(model, p[(:x,j)] >= Mxl + (xl-Mxl)*H[j,box], domain = (t >= T[j] && t <= T[j+1]))
        @polyconstraint(model, p[(:x,j)] <= Mxu + (xu-Mxu)*H[j,box], domain = (t >= T[j] && t <= T[j+1]))
        @polyconstraint(model, p[(:y,j)] >= Myl + (yl-Myl)*H[j,box], domain = (t >= T[j] && t <= T[j+1]))
        @polyconstraint(model, p[(:y,j)] <= Myu + (yu-Myu)*H[j,box], domain = (t >= T[j] && t <= T[j+1]))
    end
end

for axis in (:x,:y)
    @constraint(model, p[(axis,1)]([Tmin], [t]) == X₀[axis])
    @constraint(model, differentiate(p[(axis,1)], t)([Tmin], [t]) == X₀′[axis])
    @constraint(model, differentiate(p[(axis,1)], t, 2)([Tmin], [t]) == X₀′′[axis])

    for j in 1:N-1
        @constraint(model, p[(axis,j)]([T[j+1]], [t]) == p[(axis,j+1)]([T[j+1]], [t]))
        @constraint(model, differentiate(p[(axis,j)], t )([T[j+1]], [t]) == differentiate(p[(axis,j+1)], t)([T[j+1]], [t]))
        @constraint(model, differentiate(p[(axis,j)], t, 2)([T[j+1]], [t]) == differentiate(p[(axis,j+1)], t, 2)([T[j+1]], [t]))
    end

    @constraint(model, p[(axis,N)]([Tmax], [t]) == X₁[axis])
end

@variable(model, γ[keys(p)] ≥ 0)
for (key,val) in p
    @constraint(model, γ[key] ≥ norm(differentiate(val, t, 3)))
end
@objective(model, Min, sum(γ))

solve(model)

# Interpret solution: eval_poly(r) gives the trajectory location at time r

PP = Dict(key => getvalue(p[key]) for key in keys(p))
HH = getvalue(H)

function eval_poly(r)
    for i in 1:N
        if T[i] <= r <= T[i+1]
            return PP[(:x,i)]([r], [t]), PP[(:y,i)]([r], [t])
            break
        end
    end
    error("Time $r out of interval [$(minimum(T)),$(maximum(T))]")
end


