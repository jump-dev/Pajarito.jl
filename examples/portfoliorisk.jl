# Management of a set of portfolios with overlapping stocks and constraints on different risk measures
# Choose how much to invest in each of a limited number of stocks, to maximize payoff
using Convex, Pajarito


# Set up Convex.jl model, solve, print solution
# S is total number of stocks
# Sa is allowed number of stocks over all portfolios
# r is return on stocks
function portfoliorisk(S)
    x = Variable(S, Positive())         # Proportion to invest in each stock
    y = Variable(S, Positive(), :Bin)   # Indicators for nonzero investment in each stock

    # Maximize returns subject to x under simplex (simulates riskless asset) and cardinality of x no larger than Sa
    P = maximize(dot(r, x)
        sum(x) <= 1,
        x <= y,
        sum(y) <= Sa)

    # Add L_1/L_inf risk constraints (linear)
    for pf in pf_L1_Linf
        P.constraints += ...L1
        P.constraints += ...Linf
    end

    # Add L2 risk constraints (SOC)
    for pf in pf_L2
        P.constraints += ...
    end

    # Add robust norm-2 risk constraints (SDP)
    for pf in pf_robL2
        P.constraints += ...
    end

    # Add entropic risk constraints (Exp)
    for pf in pf_entr
        P.constraints += ...
    end

    # @show conic_problem(P)
    solve!(P, solver)

    println("\nReturns (obj) = $(P.optval)")
    @printf "\nStock  Fraction"
    for s in 1:S
        if y[s].value > 0.1
            @printf " %4d  %7.3f\n" s x[s].value
        end
    end
end
