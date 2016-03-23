#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
using ConicNonlinearBridge

include("solver.jl")
include("nonlinear.jl")
include("conic.jl")

end # module
