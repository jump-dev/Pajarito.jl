#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runsdptests(mip_solver_drives, mip_solver, sdp_solver)
    algorithm = mip_solver_drives ? "BC" : "OA"

    facts("Rotated SOC problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(sdp_solver))") do
            problem = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=sdp_solver,log_level=0))

            c = [-3.0; 0.0; 0.0;0.0]
            A = zeros(4,4)
            A[1,1] = 1.0
            A[2,2] = 1.0
            A[3,3] = 1.0
            A[4,1] = 1.0
            A[4,4] = -1.0
            b = [10.0; 3.0/2.0; 3.0; 0.0]

            constr_cones = Any[]
            push!(constr_cones, (:NonNeg, [1;2;3]))
            push!(constr_cones, (:Zero, [4]))

            var_cones = Any[]
            push!(var_cones, (:SOCRotated, [2;3;1]))
            push!(var_cones, (:Free, [4]))

            vartypes = [:Cont; :Cont; :Cont; :Int]

            MathProgBase.loadproblem!(problem, c, A, b, constr_cones, var_cones)
            MathProgBase.setvartype!(problem, vartypes)
            MathProgBase.optimize!(problem)

            @fact MathProgBase.getobjval(problem) --> roughly(-9.0, TOL)
        end
    end

    facts("Maximization problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(sdp_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1, Convex.Positive())
            z = Convex.Semidefinite(2)

            problem = Convex.maximize(3x + y,
                                x >= 0,
                                3x + 2y <= 10,
                                x^2 <= 4,
                                y >= z[2,2])

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=sdp_solver,log_level=0))

            @fact problem.optval --> roughly(8.0, TOL)
        end
    end
end
