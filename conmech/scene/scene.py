from typing import List, Optional

import jax.numpy as jnp
import numba
import numpy as np

from conmech.dynamics.dynamics import DynamicsConfiguration, SolverMatrices
from conmech.dynamics.factory.dynamics_factory_method import ConstMatrices
from conmech.helpers import lnh, nph
from conmech.helpers.lnh import get_in_base
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.body_forces import BodyForces
from conmech.scene.energy_functions import EnergyObstacleArguments
from conmech.state.body_position import BodyPosition


@numba.njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i, boundary_node in enumerate(boundary_nodes):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


class Scene(BodyForces):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        create_in_subprocess: bool,
        with_schur: bool = False,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            dynamics_config=DynamicsConfiguration(
                create_in_subprocess=create_in_subprocess,
                with_lhs=False,
                with_schur=with_schur,
            ),
        )
        self.obstacle_prop = obstacle_prop
        self.closest_obstacle_indices = None
        self.linear_obstacles: np.ndarray = np.array([[], []])
        self.mesh_obstacles: List[BodyPosition] = []
        self.energy_functions = None
        self.clear()

    def prepare(self, inner_forces):
        super().prepare(inner_forces)
        if not self.has_no_obstacles:
            self.closest_obstacle_indices = get_closest_obstacle_to_boundary_numba(
                self.boundary_nodes, self.obstacle_nodes
            )

    def normalize_and_set_obstacles(
        self,
        obstacles_unnormalized: Optional[np.ndarray],
        all_mesh_prop: Optional[List[MeshProperties]],
    ):
        if obstacles_unnormalized is not None and obstacles_unnormalized.size > 0:
            self.linear_obstacles = obstacles_unnormalized
            self.linear_obstacles[0, ...] = nph.normalize_euclidean_numba(
                self.linear_obstacles[0, ...]
            )
        if all_mesh_prop is not None:
            self.mesh_obstacles.extend(
                [BodyPosition(mesh_prop=mesh_prop, schedule=None) for mesh_prop in all_mesh_prop]
            )

    def get_energy_obstacle_args_for_jax(self, energy_functions, temperature=None):
        _ = temperature

        displacement = self.normalized_displacement_old
        velocity = self.normalized_velocity_old

        base_displacement = displacement + self.time_step * velocity
        body_prop = self.body_prop.get_tuple()

        rhs_acceleration = self.get_normalized_integrated_forces_column_for_jax()
        if temperature is not None:
            rhs_acceleration += jnp.array(self.matrices.thermal_expansion.T @ temperature)

        args = EnergyObstacleArguments(
            lhs_acceleration_jax=self.solver_cache.lhs_acceleration_jax,
            rhs_acceleration=rhs_acceleration,
            boundary_velocity_old=jnp.asarray(self.norm_boundary_velocity_old),
            boundary_normals=jnp.asarray(self.get_normalized_boundary_normals()),
            boundary_obstacle_normals=jnp.asarray(self.get_norm_boundary_obstacle_normals()),
            penetration=jnp.asarray(self.get_penetration_scalar()),
            surface_per_boundary_node=jnp.asarray(self.get_surface_per_boundary_node()),
            body_prop=body_prop,
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
            base_displacement=base_displacement,
            element_initial_volume=self.matrices.element_initial_volume,
            dx_big_jax=self.matrices.dx_big_jax,
            base_energy_displacement=energy_functions.compute_displacement_energy_jax(
                displacement=base_displacement,
                dx_big_jax=self.matrices.dx_big_jax,
                element_initial_volume=self.matrices.element_initial_volume,
                body_prop=body_prop,
            ),
            base_velocity=velocity,
            base_energy_velocity=energy_functions.compute_velocity_energy_jax(
                velocity=velocity,
                dx_big_jax=self.matrices.dx_big_jax,
                element_initial_volume=self.matrices.element_initial_volume,
                body_prop=body_prop,
            ),
        )
        return args

    @property
    def linear_obstacle_nodes(self):
        return self.linear_obstacles[1, ...]

    @property
    def linear_obstacle_normals(self):
        return self.linear_obstacles[0, ...]

    @property
    def obstacle_nodes(self):
        all_nodes = []
        all_nodes.extend(list(self.linear_obstacle_nodes))
        all_nodes.extend([m.boundary_nodes for m in self.mesh_obstacles])
        return np.vstack(all_nodes)

    def get_obstacle_normals(self):
        all_normals = []
        all_normals.extend(list(self.linear_obstacle_normals))
        all_normals.extend([m.get_boundary_normals() for m in self.mesh_obstacles])
        return np.vstack(all_normals)

    @property
    def boundary_obstacle_nodes(self):
        return self.obstacle_nodes[self.closest_obstacle_indices]

    @property
    def norm_boundary_obstacle_nodes(self):
        return self.normalize_rotate(self.boundary_obstacle_nodes - self.mean_moved_nodes)

    def get_norm_obstacle_normals(self):
        return self.normalize_rotate(self.get_obstacle_normals())

    def get_boundary_obstacle_normals(self):
        return self.get_obstacle_normals()[self.closest_obstacle_indices]

    def get_norm_boundary_obstacle_normals(self):
        return self.normalize_rotate(self.get_boundary_obstacle_normals())

    @property
    def normalized_obstacle_nodes(self):
        return self.normalize_rotate(self.obstacle_nodes - self.mean_moved_nodes)

    @property
    def boundary_velocity_old(self):
        return self.velocity_old[self.boundary_indices]

    @property
    def norm_boundary_velocity_old(self):
        return self.normalized_velocity_old[self.boundary_indices]

    @property
    def normalized_boundary_nodes(self):
        return self.normalized_nodes[self.boundary_indices]

    def get_penetration_scalar(self):
        return (-1) * nph.elementwise_dot(
            (self.normalized_boundary_nodes - self.norm_boundary_obstacle_nodes),
            self.get_norm_boundary_obstacle_normals(),
        ).reshape(-1, 1)

    def get_penetration_positive(self):
        penetration = self.get_penetration_scalar()
        return penetration * (penetration > 0)

    def __get_boundary_penetration(self):
        return self.get_penetration_positive() * self.get_boundary_obstacle_normals()

    def get_normalized_boundary_penetration(self):
        return self.normalize_rotate(self.__get_boundary_penetration())

    def __get_boundary_v_tangential(self):
        return nph.get_tangential(self.boundary_velocity_old, self.get_boundary_normals())

    def __get_normalized_boundary_v_tangential(self):
        return nph.get_tangential_numba(
            self.norm_boundary_velocity_old, self.get_normalized_boundary_normals()
        )

    def __get_friction_vector(self):
        return (self.get_penetration_scalar() > 0) * np.nan_to_num(
            nph.normalize_euclidean_numba(self.__get_normalized_boundary_v_tangential())
        )

    def get_normal_response_input(self):
        return (
            self.obstacle_prop.hardness * self.get_penetration_positive()
        )  # self.get_normalized_boundary_penetration()

    def get_friction_input(self):
        return self.obstacle_prop.friction * self.__get_friction_vector()

    def get_resistance_normal(self):
        return _obstacle_resistance_potential_normal(
            self.get_penetration_positive(), self.obstacle_prop.hardness, self.time_step
        )

    def get_resistance_tangential(self):
        return _obstacle_resistance_potential_tangential(
            self.get_penetration_positive(),
            self.__get_boundary_v_tangential(),
            self.obstacle_prop.friction,
            self.time_step,
        )

    @staticmethod
    def complete_mesh_boundary_data_with_zeros(mesh: Mesh, data: np.ndarray):
        return np.pad(data, ((0, mesh.nodes_count - len(data)), (0, 0)), "constant")

    def complete_boundary_data_with_zeros(self, data: np.ndarray):
        return Scene.complete_mesh_boundary_data_with_zeros(self, data)

    @property
    def has_no_obstacles(self):
        return self.linear_obstacles.size == 0 and len(self.mesh_obstacles) == 0

    def _get_colliding_nodes_indicator(self):
        if self.has_no_obstacles:
            return np.zeros((self.nodes_count, 1), dtype=np.int64)
        return self.complete_boundary_data_with_zeros((self.get_penetration_scalar() > 0) * 1)

    def is_colliding(self):
        return np.any(self._get_colliding_nodes_indicator())

    def prepare_to_save(self):
        self.energy_functions = None
        self.matrices = ConstMatrices()
        # lhs_sparse = self.solver_cache.lhs_sparse
        self.solver_cache = SolverMatrices()
        # self.solver_cache.lhs_sparse = lhs_sparse
        # self.reduced ...

    @property
    @Mesh.normalization_decorator
    def normalized_exact_acceleration(self):
        return self.normalize_rotate(self.exact_acceleration)

    @property
    @Mesh.normalization_decorator
    def normalized_lifted_acceleration(self):
        return self.normalize_rotate(self.lifted_acceleration)

    @Mesh.normalization_decorator
    def force_denormalize(self, acceleration):
        return self.denormalize_rotate(acceleration)

    @property
    def normalized_exact_new_displacement(self):
        return self.to_normalized_displacement(self.exact_acceleration)

    @property
    def normalized_lifted_new_displacement(self):
        return self.to_normalized_displacement(self.lifted_acceleration)

    def to_displacement(self, acceleration):
        velocity_new = self.velocity_old + self.time_step * acceleration
        displacement_new = self.displacement_old + self.time_step * velocity_new
        return displacement_new

    def from_displacement(self, displacement):
        velocity = (displacement - self.displacement_old) / self.time_step
        acceleration = (velocity - self.velocity_old) / self.time_step
        return acceleration

    @Mesh.normalization_decorator
    def to_normalized_displacement(self, acceleration):
        displacement_new = self.to_displacement(acceleration)

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(moved_nodes_new, axis=0)),
            self.get_rotation(displacement_new),
        )
        return new_normalized_nodes - self.normalized_initial_nodes

    @Mesh.normalization_decorator
    def to_normalized_displacement_rotated(self, acceleration):
        displacement_new = self.to_displacement(acceleration)

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(moved_nodes_new, axis=0)),
            self.get_rotation(self.displacement_old),
        )
        assert np.allclose(new_normalized_nodes, self.normalize_shift_and_rotate(moved_nodes_new))
        return new_normalized_nodes - self.normalized_initial_nodes

    @Mesh.normalization_decorator
    def to_normalized_displacement_rotated_displaced(self, acceleration):
        displacement_new = self.to_displacement(acceleration)

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(self.moved_nodes, axis=0)),
            self.get_rotation(self.displacement_old),
        )
        assert np.allclose(
            new_normalized_nodes,
            self.normalize_rotate(moved_nodes_new - np.mean(self.moved_nodes, axis=0)),
        )
        return new_normalized_nodes - self.normalized_initial_nodes

    def get_centered_nodes(self, displacement):
        nodes = self.centered_initial_nodes + displacement
        centered_nodes = lnh.get_in_base(
            (nodes - np.mean(nodes, axis=0)), self.get_rotation(displacement)
        )
        return centered_nodes

    def get_displacement(self, base, position, base_displacement=None):
        if base_displacement is None:
            centered_nodes = self.centered_nodes
        else:
            centered_nodes = self.get_centered_nodes(base_displacement)
        moved_centered_nodes = lnh.get_in_base(centered_nodes, np.linalg.inv(base)) + position
        displacement = moved_centered_nodes - self.centered_initial_nodes
        return displacement
