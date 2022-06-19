# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# MathOptInterface copy_to implementation

function MOI.copy_to(opt::Optimizer, src::MOI.ModelLike)
    idx_map = MOI.Utilities.IndexMap()
    get_src_cons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())

    # variable attributes including variable warm start
    has_warm_start = false
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if attr == MOI.VariableName()
            continue
        elseif attr == MOI.VariablePrimalStart()
            has_warm_start = true
            continue
        end
        throw(MOI.UnsupportedAttribute(attr))
    end
    function get_start(j::Int)
        return something(MOI.get(src, MOI.VariablePrimalStart(), VI(j)), NaN)
    end

    # variables
    n = MOI.get(src, MOI.NumberOfVariables())
    opt.warm_start = has_warm_start ? fill(NaN, n) : Float64[]
    j = 0
    # integer variables
    cis = get_src_cons(VI, MOI.Integer)
    opt.num_int_vars = length(cis)
    for ci in cis
        j += 1
        idx_map[ci] = ci
        idx_map[VI(ci.value)] = VI(j)
        if has_warm_start
            opt.warm_start[j] = get_start(j)
        end
    end
    # continuous variables
    for vj in MOI.get(src, MOI.ListOfVariableIndices())
        haskey(idx_map, vj) && continue
        j += 1
        idx_map[vj] = VI(j)
        if has_warm_start
            opt.warm_start[j] = get_start(j)
        end
    end
    @assert j == n

    # objective
    opt.obj_sense = MOI.MIN_SENSE
    model_c = zeros(n)
    obj_offset = 0.0
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr == MOI.Name()
            continue
        elseif attr == MOI.ObjectiveSense()
            opt.obj_sense = MOI.get(src, MOI.ObjectiveSense())
        elseif attr isa MOI.ObjectiveFunction
            F = MOI.get(src, MOI.ObjectiveFunctionType())
            if !(F <: Union{VI,SAF})
                error("objective function type $F not supported")
            end
            obj = convert(SAF, MOI.get(src, MOI.ObjectiveFunction{F}()))
            for t in obj.terms
                model_c[idx_map[t.variable].value] += t.coefficient
            end
            obj_offset = obj.constant
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    if opt.obj_sense == MOI.MAX_SENSE
        model_c .*= -1
        obj_offset *= -1
    end
    opt.obj_offset = obj_offset
    opt.c = model_c

    # constraints
    get_con_fun(con_idx) = MOI.get(src, MOI.ConstraintFunction(), con_idx)
    get_con_set(con_idx) = MOI.get(src, MOI.ConstraintSet(), con_idx)

    # SOS1/2 constraints
    opt.SOS12_cons = Pair{Vector{Int},<:SOS12}[]
    for S in (MOI.SOS1{Float64}, MOI.SOS2{Float64}), ci in get_src_cons(VV, S)
        fi = get_con_fun(ci)
        si = get_con_set(ci)
        idxs = [idx_map[vi].value for vi in fi.variables]
        push!(opt.SOS12_cons, (idxs => si))
        idx_map[ci] = ci
    end

    # equality constraints
    (IA, JA, VA) = (Int[], Int[], Float64[])
    model_b = Float64[]
    for F in (VV, VAF), ci in get_src_cons(F, MOI.Zeros)
        fi = get_con_fun(ci)
        _constraint_IJV(IA, JA, VA, model_b, fi, idx_map)
        idx_map[ci] = ci
    end
    opt.A = SparseArrays.dropzeros!(
        SparseArrays.sparse(IA, JA, VA, length(model_b), n),
    )
    opt.b = model_b

    # conic constraints
    (IG, JG, VG) = (Int[], Int[], Float64[])
    model_h = Float64[]
    cones = MOI.AbstractVectorSet[]
    cone_idxs = Vector{UnitRange{Int}}()

    # build up one nonnegative cone
    for F in (VV, VAF), ci in get_src_cons(F, MOI.Nonnegatives)
        fi = get_con_fun(ci)
        _constraint_IJV(IG, JG, VG, model_h, fi, idx_map)
        idx_map[ci] = ci
    end
    if !isempty(model_h)
        q = length(model_h)
        push!(cones, MOI.Nonnegatives(q))
        push!(cone_idxs, 1:q)
    end

    # other conic constraints
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(opt, F, S)
            throw(MOI.UnsupportedConstraint{F,S}())
        end
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            if attr == MOI.ConstraintName() ||
               attr == MOI.ConstraintPrimalStart() ||
               attr == MOI.ConstraintDualStart()
                continue
            end
            throw(MOI.UnsupportedAttribute(attr))
        end
        if S in (MOI.Zeros, MOI.Nonnegatives, MOI.Integer) || S <: SOS12
            continue # already copied these constraints
        end

        for ci in get_src_cons(F, S)
            fi = get_con_fun(ci)
            idxs = _constraint_IJV(IG, JG, VG, model_h, fi, idx_map)
            push!(cone_idxs, idxs)
            si = get_con_set(ci)
            push!(cones, si)
            idx_map[ci] = ci
        end
    end

    G = SparseArrays.sparse(IG, JG, VG, length(model_h), length(model_c))
    opt.G = SparseArrays.dropzeros!(G)
    opt.h = model_h
    opt.cones = cones
    opt.cone_idxs = cone_idxs
    opt.incumbent = fill(NaN, length(model_c))

    return idx_map
end

function _constraint_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector,
    vect::Vector,
    func::VV,
    idx_map::MOI.IndexMap,
)
    dim = MOI.output_dimension(func)
    idxs = length(vect) .+ (1:dim)
    append!(vect, 0.0 for _ in 1:dim)
    append!(IM, idxs)
    append!(JM, idx_map[vi].value for vi in func.variables)
    append!(VM, -1.0 for _ in 1:dim)
    return idxs
end

function _constraint_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector,
    vect::Vector,
    func::VAF,
    idx_map::MOI.IndexMap,
)
    dim = MOI.output_dimension(func)
    start = length(vect)
    append!(vect, func.constants)
    append!(IM, start + vt.output_index for vt in func.terms)
    append!(JM, idx_map[vt.scalar_term.variable].value for vt in func.terms)
    append!(VM, -vt.scalar_term.coefficient for vt in func.terms)
    return start .+ (1:dim)
end
