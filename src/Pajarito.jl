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

import MathProgBase

include("nonlinear.jl")
include("conic.jl")

end # module
