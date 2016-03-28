#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function OAprintLevel(iter, mip_objval, conic_objval, optimality_gap, best_objval, primal_infeasibility, OA_infeasibility)

    if abs(conic_objval) == Inf || isnan(conic_objval)
        conic_objval_str = @sprintf "%s" "              " 
    else
        conic_objval_str = @sprintf "%+.7e" conic_objval
    end 
    if abs(optimality_gap) == Inf || isnan(optimality_gap)
        optimality_gap_str = @sprintf "%s" "              "
    else
        optimality_gap_str = @sprintf "%+.7e" optimality_gap
    end
    if abs(best_objval) == Inf || isnan(best_objval)
        best_objval_str = @sprintf "%s" "              " 
    else
        best_objval_str = @sprintf "%+.7e" best_objval
    end

    @printf "%9d   %+.7e   %s   %s   %s   %+.7e   %+.7e\n" iter mip_objval conic_objval_str optimality_gap_str best_objval_str primal_infeasibility OA_infeasibility
end


