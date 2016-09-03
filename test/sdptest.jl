#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runsdptests(mip_solver_drives, mip_solver, sdp_solver)
    algorithm = mip_solver_drives ? "BC" : "OA"

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

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=sdp_solver, log_level=2))

            @fact problem.optval --> roughly(8.0, TOL)
        end
    end
end
