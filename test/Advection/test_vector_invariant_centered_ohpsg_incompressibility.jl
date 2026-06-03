using Test
using Oceananigans
using Oceananigans.Advection: VectorInvariant, U_dot_∇u, U_dot_∇v
using Oceananigans.Advection: U_dot_∇u_hydrostatic_metric, U_dot_∇v_hydrostatic_metric
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: horizontal_volume_flux_div_xyᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using LinearAlgebra
using Random

const HydrostaticModels = Oceananigans.Models.HydrostaticFreeSurfaceModels

function independent_octahealpix_face_indices(grid)
    Nx, Ny, _ = size(grid)
    Nu = Nx * Ny

    u_index(i, j) = (j - 1) * Nx + i
    v_index(i, j) = Nu + (j - 1) * Nx + i
    c_index(i, j) = (j - 1) * Nx + i

    return Nx, Ny, Nu, u_index, v_index, c_index
end

function set_independent_octahealpix_velocity_state!(model, coefficients)
    grid = model.grid
    Nx, Ny, Nu, u_index, v_index, _ = independent_octahealpix_face_indices(grid)
    u = model.velocities.u
    v = model.velocities.v

    fill!(parent(u), 0)
    fill!(parent(v), 0)

    for j in 1:Ny, i in 1:Nx
        u[i, j, 1] = coefficients[u_index(i, j)]
        v[i, j, 1] = coefficients[v_index(i, j)]
    end

    fill_halo_regions!((u, v))
    HydrostaticModels.update_vertical_velocities!(model.velocities, grid, model)
    fill_halo_regions!(model.velocities)

    return nothing
end

function octahealpix_horizontal_divergence_vector(model)
    grid = model.grid
    Nx, Ny, Nu, _, _, c_index = independent_octahealpix_face_indices(grid)
    u = model.velocities.u
    v = model.velocities.v
    divergence = zeros(eltype(grid), Nu)

    for j in 1:Ny, i in 1:Nx
        divergence[c_index(i, j)] = horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid, u, v)
    end

    return divergence
end

function octahealpix_horizontal_divergence_matrix(model)
    grid = model.grid
    Nx, Ny, Nu, _, _, _ = independent_octahealpix_face_indices(grid)
    number_of_degrees_of_freedom = 2Nu
    divergence_matrix = zeros(eltype(grid), Nu, number_of_degrees_of_freedom)
    coefficients = zeros(eltype(grid), number_of_degrees_of_freedom)

    for column in 1:number_of_degrees_of_freedom
        fill!(coefficients, 0)
        coefficients[column] = 1
        set_independent_octahealpix_velocity_state!(model, coefficients)
        divergence_matrix[:, column] .= octahealpix_horizontal_divergence_vector(model)
    end

    return divergence_matrix
end

function octahealpix_vector_invariant_tendency_vector(model)
    grid = model.grid
    Nx, Ny, Nu, u_index, v_index, _ = independent_octahealpix_face_indices(grid)
    scheme = model.advection.momentum
    velocities = model.velocities
    tendency = zeros(eltype(grid), 2Nu)

    for j in 1:Ny, i in 1:Nx
        u_advection = U_dot_∇u(i, j, 1, grid, scheme, velocities) +
                      U_dot_∇u_hydrostatic_metric(i, j, 1, grid, scheme, velocities, velocities)

        v_advection = U_dot_∇v(i, j, 1, grid, scheme, velocities) +
                      U_dot_∇v_hydrostatic_metric(i, j, 1, grid, scheme, velocities, velocities)

        tendency[u_index(i, j)] = -u_advection
        tendency[v_index(i, j)] = -v_advection
    end

    return tendency
end

function centered_vi_incompressibility_defect(; N = 4, seed = 42)
    grid = SphericalShellGrid(CPU(), Float64;
                              mapping = OctaHEALPixMapping(N),
                              z = (0, 1),
                              radius = 1,
                              halo = (5, 5, 3))

    model = HydrostaticFreeSurfaceModel(grid;
                                        tracers = (),
                                        buoyancy = nothing,
                                        coriolis = nothing,
                                        free_surface = nothing,
                                        closure = nothing,
                                        momentum_advection = VectorInvariant())

    Nx, Ny, Nu, _, _, _ = independent_octahealpix_face_indices(grid)
    Random.seed!(seed)
    initial_coefficients = 1e-2 .* randn(2Nu)

    divergence_matrix = octahealpix_horizontal_divergence_matrix(model)
    divergence = divergence_matrix * initial_coefficients
    projection = divergence_matrix' * (pinv(divergence_matrix * divergence_matrix'; rtol = 1e-10) * divergence)
    projected_coefficients = initial_coefficients - projection

    set_independent_octahealpix_velocity_state!(model, projected_coefficients)

    state_divergence = octahealpix_horizontal_divergence_vector(model)
    tendency = octahealpix_vector_invariant_tendency_vector(model)
    tendency_divergence = divergence_matrix * tendency

    set_independent_octahealpix_velocity_state!(model, projected_coefficients + tendency)
    HydrostaticModels.project_rigid_lid_velocities!(model, 1)
    projected_update_divergence = octahealpix_horizontal_divergence_vector(model)

    return (; maximum_state_divergence = maximum(abs.(state_divergence)),
              maximum_tendency_divergence = maximum(abs.(tendency_divergence)),
              maximum_projected_update_divergence = maximum(abs.(projected_update_divergence)))
end

function initial_rigid_lid_projection_defect(; N = 4, seed = 1234)
    grid = SphericalShellGrid(CPU(), Float64;
                              mapping = OctaHEALPixMapping(N),
                              z = (0, 1),
                              radius = 1,
                              halo = (5, 5, 3))

    model = HydrostaticFreeSurfaceModel(grid;
                                        tracers = (),
                                        buoyancy = nothing,
                                        coriolis = nothing,
                                        free_surface = nothing,
                                        closure = nothing,
                                        momentum_advection = VectorInvariant())

    Nx, Ny, Nu, _, _, _ = independent_octahealpix_face_indices(grid)
    Random.seed!(seed)
    coefficients = 1e-2 .* randn(2Nu)

    set_independent_octahealpix_velocity_state!(model, coefficients)
    initial_divergence = octahealpix_horizontal_divergence_vector(model)
    update_state!(model)
    projected_divergence = octahealpix_horizontal_divergence_vector(model)

    return (; maximum_initial_divergence = maximum(abs.(initial_divergence)),
              maximum_projected_divergence = maximum(abs.(projected_divergence)),
              projection_iterations = model.rigid_lid_projection.conjugate_gradient_solver.iteration)
end

@testset "Centered VectorInvariant OHPSG incompressibility preservation" begin
    defect = centered_vi_incompressibility_defect()
    initial_projection_defect = initial_rigid_lid_projection_defect()

    @info "Centered VI OHPSG incompressibility defect" defect.maximum_state_divergence defect.maximum_tendency_divergence defect.maximum_projected_update_divergence
    @info "Centered VI OHPSG initial projection defect" initial_projection_defect.maximum_initial_divergence initial_projection_defect.maximum_projected_divergence initial_projection_defect.projection_iterations

    @test defect.maximum_state_divergence < 1e-12
    @test_broken defect.maximum_tendency_divergence < 1e-12
    @test defect.maximum_projected_update_divergence < 1e-10
    @test initial_projection_defect.maximum_initial_divergence > 1e-2
    @test initial_projection_defect.maximum_projected_divergence < 1e-10
    @test initial_projection_defect.projection_iterations > 0
end
