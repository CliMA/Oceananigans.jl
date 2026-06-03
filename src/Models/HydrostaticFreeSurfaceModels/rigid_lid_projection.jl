#####
##### Rigid-lid horizontal projection
#####
##### `free_surface = nothing` means a rigid lid. On non-orthogonal
##### `SphericalShellGrid`s, centered vector-invariant momentum advection must be
##### followed by a Hodge-weighted horizontal projection to keep the state in the
##### discrete rigid-lid subspace:
#####
#####     u <- u - K^{-1} D^T (D K^{-1} D^T)^{-1} D u
#####
##### where `K` is the weighted covariant-to-volume-flux Hodge map and `D` is
##### horizontal volume-flux divergence. The fallback is intentionally a no-op;
##### production specializations must not use masks, damping, or Euclidean
##### projection.

using Oceananigans.Fields: CenterField, XFaceField, YFaceField
using Oceananigans.Operators: horizontal_volume_flux_div_xyᶜᶜᶜ,
                              hodge_compatible_pressure_correction_uᶠᶜᶜ,
                              hodge_compatible_pressure_correction_vᶜᶠᶜ
using Oceananigans.Solvers: ConjugateGradientSolver, solve!
using Oceananigans.Utils: KernelParameters
using Statistics: mean

struct RigidLidProjectionSolver{S, P, R, C}
    conjugate_gradient_solver :: S
    pressure :: P
    rhs :: R
    correction :: C
end

materialize_rigid_lid_projection(grid, free_surface) = nothing

materialize_rigid_lid_projection(grid::SphericalShellGrid, free_surface::Nothing) =
    RigidLidProjectionSolver(grid)

rigid_lid_projection(model) = model.rigid_lid_projection

project_rigid_lid_velocities!(model, Δt) =
    project_rigid_lid_velocities!(model, model.rigid_lid_projection, Δt)

project_rigid_lid_velocities!(model, projection, Δt) = nothing

function project_initial_rigid_lid_velocities!(model)
    if model.clock.iteration == 0
        project_rigid_lid_velocities!(model, one(model.grid))
    end

    return nothing
end

function project_rigid_lid_velocities!(model, projection::RigidLidProjectionSolver, Δt)
    solve_rigid_lid_projection!(projection, model.velocities, model.grid)
    subtract_rigid_lid_pressure_correction!(model.velocities, projection.correction, model.grid, one(model.grid))

    return nothing
end

function RigidLidProjectionSolver(grid::SphericalShellGrid;
                                  maxiter = prod(size(grid)),
                                  reltol = eps(eltype(grid))^(3//4),
                                  abstol = eps(eltype(grid))^(3//4))
    pressure = CenterField(grid)
    rhs = CenterField(grid)
    correction = (u = XFaceField(grid), v = YFaceField(grid))

    conjugate_gradient_solver =
        ConjugateGradientSolver(rigid_lid_projection_linear_operation!;
                                template_field = pressure,
                                maxiter,
                                reltol,
                                abstol,
                                enforce_gauge_condition! = no_rigid_lid_gauge_enforcement!)

    return RigidLidProjectionSolver(conjugate_gradient_solver, pressure, rhs, correction)
end

@inline no_rigid_lid_gauge_enforcement!(pressure, residual) = nothing

function rigid_lid_projection_linear_operation!(schur_complement, pressure, grid, correction)
    compute_rigid_lid_projection_schur_complement!(schur_complement, pressure, grid, correction)
    return nothing
end

function enforce_rigid_lid_zero_mean!(field)
    grid = field.grid
    field_mean = mean(field)

    launch!(architecture(grid), grid, :xyz,
            _subtract_rigid_lid_mean!, field, field_mean)

    return nothing
end

function enforce_rigid_lid_zero_mean_gauge!(pressure, residual)
    enforce_rigid_lid_zero_mean!(pressure)
    enforce_rigid_lid_zero_mean!(residual)

    return nothing
end

@kernel function _subtract_rigid_lid_mean!(field, field_mean)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] -= field_mean
end

function solve_rigid_lid_projection!(projection::RigidLidProjectionSolver, velocities, grid::SphericalShellGrid)
    compute_rigid_lid_projection_rhs!(projection.rhs, grid, velocities)
    fill!(parent(projection.pressure), zero(eltype(grid)))

    solve!(projection.pressure,
           projection.conjugate_gradient_solver,
           projection.rhs,
           grid,
           projection.correction)

    compute_rigid_lid_pressure_correction!(projection.correction, grid, projection.pressure)

    return nothing
end

function compute_rigid_lid_projection_rhs!(rhs, grid::SphericalShellGrid, velocities)
    launch!(architecture(grid), grid, :xyz,
            _compute_rigid_lid_projection_rhs!, rhs, grid, velocities.u, velocities.v)

    return nothing
end

@kernel function _compute_rigid_lid_projection_rhs!(rhs, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v)
end

function compute_rigid_lid_pressure_correction!(correction, grid::SphericalShellGrid, pressure)
    parameters = KernelParameters(1:grid.Nx, 1:grid.Ny, 1:grid.Nz)

    launch!(architecture(grid), grid, parameters,
            _compute_rigid_lid_pressure_correction_u!, correction.u, grid, pressure)

    launch!(architecture(grid), grid, parameters,
            _compute_rigid_lid_pressure_correction_v!, correction.v, grid, pressure)

    fill_halo_regions!((correction.u, correction.v))

    return nothing
end

@kernel function _compute_rigid_lid_pressure_correction_u!(u_correction, grid, pressure)
    i, j, k = @index(Global, NTuple)
    @inbounds u_correction[i, j, k] =
        hodge_compatible_pressure_correction_uᶠᶜᶜ(i, j, k, grid, pressure)
end

@kernel function _compute_rigid_lid_pressure_correction_v!(v_correction, grid, pressure)
    i, j, k = @index(Global, NTuple)
    @inbounds v_correction[i, j, k] =
        hodge_compatible_pressure_correction_vᶜᶠᶜ(i, j, k, grid, pressure)
end

function compute_rigid_lid_projection_schur_complement!(schur_complement, pressure,
                                                        grid::SphericalShellGrid, correction)
    compute_rigid_lid_pressure_correction!(correction, grid, pressure)
    compute_rigid_lid_projection_rhs!(schur_complement, grid, correction)

    return nothing
end

function subtract_rigid_lid_pressure_correction!(velocities, correction, grid::SphericalShellGrid, scale)
    parameters = KernelParameters(1:grid.Nx, 1:grid.Ny, 1:grid.Nz)

    launch!(architecture(grid), grid, parameters,
            _subtract_rigid_lid_pressure_correction!, velocities.u, correction.u, scale)

    launch!(architecture(grid), grid, parameters,
            _subtract_rigid_lid_pressure_correction!, velocities.v, correction.v, scale)

    fill_halo_regions!((velocities.u, velocities.v))

    return nothing
end

@kernel function _subtract_rigid_lid_pressure_correction!(velocity, correction, scale)
    i, j, k = @index(Global, NTuple)
    @inbounds velocity[i, j, k] -= scale * correction[i, j, k]
end
