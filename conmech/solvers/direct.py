"""
Created at 18.02.2021
"""

import scipy.optimize
import numpy as np

from conmech.solvers.solver_methods import make_f
from conmech.solvers.solver import Solver
from conmech.solvers._solvers import Solvers


@Solvers.register("static", "direct")
class Direct(Solver):
    def __init__(
        self,
        grid,
        inner_forces,
        outer_forces,
        coefficients,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            grid,
            inner_forces,
            outer_forces,
            coefficients,
            time_step,
            contact_law,
            friction_bound,
        )

        self.f = make_f(
            jnZ=contact_law.subderivative_normal_direction,
            jtZ=contact_law.regularized_subderivative_tangential_direction,
            h=friction_bound,
        )

    def __str__(self):
        return "direct"

    def solve(self, initial_guess: np.ndarray, **kwargs) -> np.ndarray:
        result = scipy.optimize.fsolve(
            self.f,
            initial_guess,
            args=(
                self.mesh.initial_points,
                self.mesh.boundaries.contact,
                self.B,
                self.forces.Zero[: self.mesh.independent_nodes_conunt],  # TODO
                self.forces.One[: self.mesh.independent_nodes_conunt],
            ),
        )
        return np.asarray(result)
