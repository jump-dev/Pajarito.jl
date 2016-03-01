######################################################
# This package contains the mixed-integer non-linear
# programming (MINLP) problem solver Pajarito.jl:
#
#       P olyhedral
#       A pproximation
# (in)  J ulia :
#       A utomatic
#       R eformulations
# (for) I n T eger
#       O ptimization
# 
# It applies outer approximation to a series of
# mixed-integer linear programming problems
# that approximates the original MINLP in a polyhedral
# form.
######################################################

module Pajarito

using Ipopt
using CPLEX
using Compat

import MathProgBase

include("pajarito_nlp.jl")
include("pajarito_dcp.jl")
include("pajarito_feasibility.jl")
include("conic_nlp_wrapper.jl")

end # module
