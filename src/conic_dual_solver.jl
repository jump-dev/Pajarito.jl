# Wrapper which dualizes a MathProgBase conic problem before calling the conic solver
# See http://mathprogbasejl.readthedocs.io/en/latest/conic.html
#
# Strong duality must hold between the primal and dual
# NOTE: potentially returns incorrect status (eg if both infeasible)
#
# Primal:
# min_x   c^Tx
# s.t.    b - Ax   \in K_1
#         x        \in K_2
#
# Dual:
# max_y   -b^Ty
# s.t.    c + A^Ty \in K_2^*
#         y        \in K_1^*
#

export ConicDualWrapper

mutable struct ConicDualWrapper <: MathProgBase.AbstractMathProgSolver
    conicsolver::MathProgBase.AbstractMathProgSolver
end

ConicDualWrapper(;conicsolver=nothing) = ConicDualWrapper(conicsolver)


mutable struct ConicDualModel <: MathProgBase.AbstractConicModel
    conicsolver::MathProgBase.AbstractMathProgSolver
    dualmodel::MathProgBase.AbstractConicModel
    dualstatus::Symbol

    function ConicDualModel(conicsolver)
        m = new()
        m.conicsolver = conicsolver
        return m
    end
end


const conedual = Dict{Symbol,Symbol}(
    :Zero => :Free,
    :Free => :Zero,
    :NonNeg => :NonNeg,
    :NonPos => :NonPos,
    :ExpPrimal => :ExpDual,
    :SDP => :SDP,
    :SOC => :SOC,
    :SOCRotated => :SOCRotated
    )


MathProgBase.ConicModel(s::ConicDualWrapper) = ConicDualModel(s.conicsolver)

MathProgBase.supportedcones(s::ConicDualWrapper) = [:Free, :Zero, :NonNeg, :NonPos, :SOC, :SOCRotated, :ExpPrimal, :ExpDual, :SDP]

function MathProgBase.loadproblem!(m::ConicDualModel, c::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64}, concones::Vector{Tuple{Symbol,Vector{Int}}}, varcones::Vector{Tuple{Symbol,Vector{Int}}})
    if m.conicsolver == nothing
        error("Conic solver is not specified")
    end

    dualize = (coneinds -> (conedual[coneinds[1]], coneinds[2]))

    m.dualmodel = MathProgBase.ConicModel(m.conicsolver)
    MathProgBase.loadproblem!(m.dualmodel, b, -A', c, map(dualize, varcones), map(dualize, concones))
end

function MathProgBase.optimize!(m::ConicDualModel)
    MathProgBase.optimize!(m.dualmodel)
end

# function MathProgBase.setbvec!(m::ConicDualModel, newbvec::Vector{Float64})
#     MathProgBase.setcvec!(m.dualmodel, newbvec)
# end

function MathProgBase.status(m::ConicDualModel)
    dualstatus = MathProgBase.status(m.dualmodel)

    # TODO Potentially returns incorrect status, but this is good enough for our purposes
    if dualstatus == :Optimal
        return :Optimal
    elseif dualstatus == :Infeasible
        return :Unbounded
    elseif dualstatus == :Unbounded
        return :Infeasible
    else
        return :ConicFailure
    end
end

MathProgBase.getobjval(m::ConicDualModel) = -MathProgBase.getobjval(m.dualmodel)

MathProgBase.getsolution(m::ConicDualModel) = MathProgBase.getdual(m.dualmodel)

MathProgBase.getdual(m::ConicDualModel) = MathProgBase.getsolution(m.dualmodel)

function MathProgBase.freemodel!(m::ConicDualModel)
    if applicable(MathProgBase.freemodel!, m.dualmodel)
        MathProgBase.freemodel!(m.dualmodel)
    end
end
