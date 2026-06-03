using Test
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: horizontal_volume_flux_div_xyᶜᶜᶜ
using Oceananigans.Operators: δxᶜᵃᵃ, δyᵃᶜᵃ
using Oceananigans.Operators: Azᶠᶜᶜ, Azᶜᶠᶜ, covariant_to_volume_flux_uᶠᶜᶜ, covariant_to_volume_flux_vᶜᶠᶜ
using Oceananigans.Operators: covariant_gradient_xᶠᶜᶜ, covariant_gradient_yᶜᶠᶜ
using Oceananigans.Operators: hodge_compatible_volume_flux_div_xyᶜᶜᶜ
using Oceananigans.Operators: hodge_compatible_pressure_correction_uᶠᶜᶜ, hodge_compatible_pressure_correction_vᶜᶠᶜ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: RigidLidProjectionSolver, solve_rigid_lid_projection!
using LinearAlgebra
using Random

function octahealpix_projection_indices(grid)
    Nx, Ny, _ = size(grid)
    Nu = Nx * Ny

    u_index(i, j) = (j - 1) * Nx + i
    v_index(i, j) = Nu + (j - 1) * Nx + i
    c_index(i, j) = (j - 1) * Nx + i

    return Nx, Ny, Nu, u_index, v_index, c_index
end

function set_octahealpix_projection_state!(u, v, coefficients)
    grid = u.grid
    Nx, Ny, Nu, u_index, v_index, _ = octahealpix_projection_indices(grid)
    FT = eltype(grid)

    fill!(parent(u), zero(FT))
    fill!(parent(v), zero(FT))

    for j in 1:Ny, i in 1:Nx
        u[i, j, 1] = coefficients[u_index(i, j)]
        v[i, j, 1] = coefficients[v_index(i, j)]
    end

    fill_halo_regions!((u, v))

    return nothing
end

function octahealpix_projection_divergence_vector(u, v)
    grid = u.grid
    Nx, Ny, Nu, _, _, c_index = octahealpix_projection_indices(grid)
    divergence = zeros(eltype(grid), Nu)

    for j in 1:Ny, i in 1:Nx
        divergence[c_index(i, j)] = horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid, u, v)
    end

    return divergence
end

function octahealpix_projection_divergence_matrix(u, v)
    grid = u.grid
    Nx, Ny, Nu, _, _, _ = octahealpix_projection_indices(grid)
    number_of_degrees_of_freedom = 2Nu
    divergence_matrix = zeros(eltype(grid), Nu, number_of_degrees_of_freedom)
    coefficients = zeros(eltype(grid), number_of_degrees_of_freedom)

    for column in 1:number_of_degrees_of_freedom
        fill!(coefficients, 0)
        coefficients[column] = 1
        set_octahealpix_projection_state!(u, v, coefficients)
        divergence_matrix[:, column] .= octahealpix_projection_divergence_vector(u, v)
    end

    return divergence_matrix
end

function octahealpix_projection_volume_flux_divergence_vector(u, v)
    grid = u.grid
    Nx, Ny, Nu, _, _, c_index = octahealpix_projection_indices(grid)
    divergence = zeros(eltype(grid), Nu)

    for j in 1:Ny, i in 1:Nx
        divergence[c_index(i, j)] = δxᶜᵃᵃ(i, j, 1, grid, u) +
                                    δyᵃᶜᵃ(i, j, 1, grid, v)
    end

    return divergence
end

function octahealpix_projection_volume_flux_divergence_matrix(u, v)
    grid = u.grid
    Nx, Ny, Nu, _, _, _ = octahealpix_projection_indices(grid)
    number_of_degrees_of_freedom = 2Nu
    divergence_matrix = zeros(eltype(grid), Nu, number_of_degrees_of_freedom)
    coefficients = zeros(eltype(grid), number_of_degrees_of_freedom)

    for column in 1:number_of_degrees_of_freedom
        fill!(coefficients, 0)
        coefficients[column] = 1
        set_octahealpix_projection_state!(u, v, coefficients)
        divergence_matrix[:, column] .= octahealpix_projection_volume_flux_divergence_vector(u, v)
    end

    return divergence_matrix
end

function octahealpix_projection_hodge_compatible_operator_matrix(u, v)
    grid = u.grid
    Nx, Ny, Nu, _, _, c_index = octahealpix_projection_indices(grid)
    number_of_degrees_of_freedom = 2Nu
    divergence_matrix = zeros(eltype(grid), Nu, number_of_degrees_of_freedom)
    coefficients = zeros(eltype(grid), number_of_degrees_of_freedom)

    for column in 1:number_of_degrees_of_freedom
        fill!(coefficients, 0)
        coefficients[column] = 1
        set_octahealpix_projection_state!(u, v, coefficients)

        for j in 1:Ny, i in 1:Nx
            divergence_matrix[c_index(i, j), column] =
                hodge_compatible_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid, u, v)
        end
    end

    return divergence_matrix
end

function octahealpix_projection_hodge_weights(grid)
    Nx, Ny, Nu, u_index, v_index, _ = octahealpix_projection_indices(grid)
    weights = zeros(eltype(grid), 2Nu)

    for j in 1:Ny, i in 1:Nx
        weights[u_index(i, j)] = Azᶠᶜᶜ(i, j, 1, grid) / 2
        weights[v_index(i, j)] = Azᶜᶠᶜ(i, j, 1, grid) / 2
    end

    return weights
end

function octahealpix_projection_hodge_matrix(u, v)
    grid = u.grid
    Nx, Ny, Nu, u_index, v_index, _ = octahealpix_projection_indices(grid)
    number_of_degrees_of_freedom = 2Nu
    hodge_matrix = zeros(eltype(grid), number_of_degrees_of_freedom, number_of_degrees_of_freedom)
    coefficients = zeros(eltype(grid), number_of_degrees_of_freedom)

    for column in 1:number_of_degrees_of_freedom
        fill!(coefficients, 0)
        coefficients[column] = 1
        set_octahealpix_projection_state!(u, v, coefficients)

        for j in 1:Ny, i in 1:Nx
            hodge_matrix[u_index(i, j), column] = covariant_to_volume_flux_uᶠᶜᶜ(i, j, 1, grid, u, v)
            hodge_matrix[v_index(i, j), column] = covariant_to_volume_flux_vᶜᶠᶜ(i, j, 1, grid, u, v)
        end
    end

    return hodge_matrix
end

function octahealpix_projection_hodge_energy_matrix(u, v)
    grid = u.grid
    weights = octahealpix_projection_hodge_weights(grid)
    hodge_matrix = octahealpix_projection_hodge_matrix(u, v)
    weighted_hodge = Diagonal(weights) * hodge_matrix

    return Symmetric((weighted_hodge + weighted_hodge') / 2)
end

function octahealpix_projection_boundary_hodge_matrices(u, v)
    grid = u.grid
    Nx, Ny, Nu, _, _, _ = octahealpix_projection_indices(grid)
    number_of_degrees_of_freedom = 2Nu
    east_boundary_hodge_matrix = zeros(eltype(grid), Ny, number_of_degrees_of_freedom)
    north_boundary_hodge_matrix = zeros(eltype(grid), Nx, number_of_degrees_of_freedom)
    coefficients = zeros(eltype(grid), number_of_degrees_of_freedom)

    for column in 1:number_of_degrees_of_freedom
        fill!(coefficients, 0)
        coefficients[column] = 1
        set_octahealpix_projection_state!(u, v, coefficients)

        for j in 1:Ny
            east_boundary_hodge_matrix[j, column] = covariant_to_volume_flux_uᶠᶜᶜ(Nx + 1, j, 1, grid, u, v)
        end

        for i in 1:Nx
            north_boundary_hodge_matrix[i, column] = covariant_to_volume_flux_vᶜᶠᶜ(i, Ny + 1, 1, grid, u, v)
        end
    end

    return east_boundary_hodge_matrix, north_boundary_hodge_matrix
end

function octahealpix_projection_hodge_compatible_divergence_matrix(grid, hodge_matrix,
                                                                   east_boundary_hodge_matrix,
                                                                   north_boundary_hodge_matrix)
    Nx, Ny, Nu, u_index, v_index, c_index = octahealpix_projection_indices(grid)
    east_boundary_flux_matrix = east_boundary_hodge_matrix / hodge_matrix
    north_boundary_flux_matrix = north_boundary_hodge_matrix / hodge_matrix
    divergence_matrix = zeros(eltype(grid), Nu, 2Nu)

    for j in 1:Ny, i in 1:Nx
        row = c_index(i, j)
        divergence_matrix[row, u_index(i, j)] -= 1
        divergence_matrix[row, v_index(i, j)] -= 1

        if i == Nx
            divergence_matrix[row, :] .+= @view east_boundary_flux_matrix[j, :]
        else
            divergence_matrix[row, u_index(i + 1, j)] += 1
        end

        if j == Ny
            divergence_matrix[row, :] .+= @view north_boundary_flux_matrix[i, :]
        else
            divergence_matrix[row, v_index(i, j + 1)] += 1
        end
    end

    return divergence_matrix, east_boundary_flux_matrix, north_boundary_flux_matrix
end

function octahealpix_projection_boundary_flux_ratio_matrices(grid, hodge_matrix,
                                                             east_boundary_hodge_matrix,
                                                             north_boundary_hodge_matrix)
    Nx, Ny, Nu, u_index, v_index, _ = octahealpix_projection_indices(grid)
    east_boundary_flux_matrix = zeros(eltype(grid), Ny, 2Nu)
    north_boundary_flux_matrix = zeros(eltype(grid), Nx, 2Nu)

    for j in 1:Ny
        source_kind, source_i, source_j, _ =
            Oceananigans.Grids.octahealpix_covariant_xface_halo_source(Nx + 1, j, Nx, Ny, grid.connectivity)

        source_row = source_kind == 1 ? u_index(source_i, source_j) : v_index(source_i, source_j)
        hodge_ratio = east_boundary_hodge_matrix[j, source_row] / hodge_matrix[source_row, source_row]
        east_boundary_flux_matrix[j, source_row] = hodge_ratio
    end

    for i in 1:Nx
        source_kind, source_i, source_j, _ =
            Oceananigans.Grids.octahealpix_covariant_yface_halo_source(i, Ny + 1, Nx, Ny, grid.connectivity)

        source_row = source_kind == 1 ? u_index(source_i, source_j) : v_index(source_i, source_j)
        hodge_ratio = north_boundary_hodge_matrix[i, source_row] / hodge_matrix[source_row, source_row]
        north_boundary_flux_matrix[i, source_row] = hodge_ratio
    end

    return east_boundary_flux_matrix, north_boundary_flux_matrix
end

function hodge_weighted_divergence_free_projection(defect, divergence_matrix, hodge_energy_matrix)
    K_factorization = cholesky(hodge_energy_matrix)
    K⁻¹Dᵀ = K_factorization \ divergence_matrix'
    projection_laplacian = divergence_matrix * K⁻¹Dᵀ
    pressure = pinv(projection_laplacian; rtol = 1e-10) * (divergence_matrix * defect)

    return defect - K⁻¹Dᵀ * pressure
end

function octahealpix_projection_covariant_gradient_vector(grid, pressure_data)
    Nx, Ny, Nu, u_index, v_index, c_index = octahealpix_projection_indices(grid)
    pressure = CenterField(grid)
    gradient = zeros(eltype(grid), 2Nu)

    fill!(parent(pressure), 0)

    for j in 1:Ny, i in 1:Nx
        pressure[i, j, 1] = pressure_data[c_index(i, j)]
    end

    fill_halo_regions!(pressure)

    for j in 1:Ny, i in 1:Nx
        gradient[u_index(i, j)] = covariant_gradient_xᶠᶜᶜ(i, j, 1, grid, pressure)
        gradient[v_index(i, j)] = covariant_gradient_yᶜᶠᶜ(i, j, 1, grid, pressure)
    end

    return gradient
end

function octahealpix_projection_pressure_field(grid, pressure_data)
    Nx, Ny, _, _, _, c_index = octahealpix_projection_indices(grid)
    pressure = CenterField(grid)

    fill!(parent(pressure), 0)

    for j in 1:Ny, i in 1:Nx
        pressure[i, j, 1] = pressure_data[c_index(i, j)]
    end

    fill_halo_regions!(pressure)

    return pressure
end

function octahealpix_projection_hodge_compatible_pressure_correction_vector(grid, pressure_data)
    Nx, Ny, Nu, u_index, v_index, _ = octahealpix_projection_indices(grid)
    pressure = octahealpix_projection_pressure_field(grid, pressure_data)
    correction = zeros(eltype(grid), 2Nu)

    for j in 1:Ny, i in 1:Nx
        correction[u_index(i, j)] = hodge_compatible_pressure_correction_uᶠᶜᶜ(i, j, 1, grid, pressure)
        correction[v_index(i, j)] = hodge_compatible_pressure_correction_vᶜᶠᶜ(i, j, 1, grid, pressure)
    end

    return correction
end

function octahealpix_projection_face_vector(u, v)
    grid = u.grid
    Nx, Ny, Nu, u_index, v_index, _ = octahealpix_projection_indices(grid)
    vector = zeros(eltype(grid), 2Nu)

    for j in 1:Ny, i in 1:Nx
        vector[u_index(i, j)] = u[i, j, 1]
        vector[v_index(i, j)] = v[i, j, 1]
    end

    return vector
end

function octahealpix_projection_hodge_compatible_schur_complement_vector(u, v, pressure_data)
    grid = u.grid
    Nx, Ny, Nu, _, _, c_index = octahealpix_projection_indices(grid)
    pressure_correction = octahealpix_projection_hodge_compatible_pressure_correction_vector(grid, pressure_data)
    schur_complement = zeros(eltype(grid), Nu)

    set_octahealpix_projection_state!(u, v, pressure_correction)

    for j in 1:Ny, i in 1:Nx
        schur_complement[c_index(i, j)] =
            horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid, u, v)
    end

    return schur_complement
end

@testset "OctaHEALPix Hodge-weighted divergence-free projection" begin
    grid = SphericalShellGrid(CPU(), Float64;
                              mapping = OctaHEALPixMapping(4),
                              z = (0, 1),
                              radius = 1,
                              halo = (5, 5, 3))

    u = XFaceField(grid)
    v = YFaceField(grid)
    _, _, Nu, _, _, _ = octahealpix_projection_indices(grid)

    divergence_matrix = octahealpix_projection_divergence_matrix(u, v)
    volume_flux_divergence_matrix = octahealpix_projection_volume_flux_divergence_matrix(u, v)
    hodge_compatible_operator_matrix = octahealpix_projection_hodge_compatible_operator_matrix(u, v)
    hodge_matrix = octahealpix_projection_hodge_matrix(u, v)
    hodge_energy_matrix = octahealpix_projection_hodge_energy_matrix(u, v)
    hodge_weights = octahealpix_projection_hodge_weights(grid)
    exact_volume_flux_divergence_matrix = divergence_matrix / hodge_matrix
    east_boundary_hodge_matrix, north_boundary_hodge_matrix =
        octahealpix_projection_boundary_hodge_matrices(u, v)
    hodge_compatible_divergence_matrix, east_boundary_flux_matrix, north_boundary_flux_matrix =
        octahealpix_projection_hodge_compatible_divergence_matrix(grid, hodge_matrix,
                                                                 east_boundary_hodge_matrix,
                                                                 north_boundary_hodge_matrix)
    ratio_east_boundary_flux_matrix, ratio_north_boundary_flux_matrix =
        octahealpix_projection_boundary_flux_ratio_matrices(grid, hodge_matrix,
                                                           east_boundary_hodge_matrix,
                                                           north_boundary_hodge_matrix)
    exact_volume_flux_divergence_row_nonzeros = vec(sum(abs.(exact_volume_flux_divergence_matrix) .> 1e-10, dims=2))
    east_boundary_flux_row_nonzeros = vec(sum(abs.(east_boundary_flux_matrix) .> 1e-10, dims=2))
    north_boundary_flux_row_nonzeros = vec(sum(abs.(north_boundary_flux_matrix) .> 1e-10, dims=2))

    Random.seed!(42)
    initial_state = 1e-2 .* randn(2Nu)
    projected_state = hodge_weighted_divergence_free_projection(initial_state, divergence_matrix, hodge_energy_matrix)

    initial_divergence = divergence_matrix * initial_state
    projected_divergence = divergence_matrix * projected_state
    initial_energy = initial_state' * hodge_energy_matrix * initial_state
    projected_energy = projected_state' * hodge_energy_matrix * projected_state

    @info "OctaHEALPix Hodge projection" maximum(abs.(initial_divergence)) maximum(abs.(projected_divergence)) initial_energy projected_energy

    @test maximum(abs.(initial_divergence)) > 1e-8
    @test maximum(abs.(projected_divergence)) < 1e-12
    @test projected_energy ≤ initial_energy * (1 + 100eps(Float64))

    pressure = randn(Nu)
    pressure .-= sum(pressure) / length(pressure)

    K_factorization = cholesky(hodge_energy_matrix)
    hodge_weighted_pressure_correction = K_factorization \ divergence_matrix' * pressure
    exact_weighted_adjoint_correction = (exact_volume_flux_divergence_matrix' * pressure) ./ hodge_weights
    hodge_compatible_source_pressure_correction =
        octahealpix_projection_hodge_compatible_pressure_correction_vector(grid, pressure)
    dense_schur_complement = divergence_matrix * hodge_weighted_pressure_correction
    hodge_compatible_schur_complement =
        octahealpix_projection_hodge_compatible_schur_complement_vector(u, v, pressure)
    weighted_divergence_adjoint_correction = (volume_flux_divergence_matrix' * pressure) ./ hodge_weights
    local_covariant_gradient = octahealpix_projection_covariant_gradient_vector(grid, pressure)

    relative_exact_adjoint_error = norm(hodge_weighted_pressure_correction - exact_weighted_adjoint_correction) /
                                   norm(hodge_weighted_pressure_correction)

    relative_hodge_compatible_pressure_correction_error =
        norm(hodge_weighted_pressure_correction - hodge_compatible_source_pressure_correction) /
        norm(hodge_weighted_pressure_correction)

    relative_hodge_compatible_schur_complement_error =
        norm(dense_schur_complement - hodge_compatible_schur_complement) /
        norm(dense_schur_complement)

    relative_hodge_compatible_divergence_error =
        norm(exact_volume_flux_divergence_matrix - hodge_compatible_divergence_matrix) /
        norm(exact_volume_flux_divergence_matrix)

    relative_hodge_compatible_operator_error =
        norm(exact_volume_flux_divergence_matrix - hodge_compatible_operator_matrix) /
        norm(exact_volume_flux_divergence_matrix)

    relative_boundary_flux_ratio_error =
        (norm(east_boundary_flux_matrix - ratio_east_boundary_flux_matrix) +
         norm(north_boundary_flux_matrix - ratio_north_boundary_flux_matrix)) /
        (norm(east_boundary_flux_matrix) + norm(north_boundary_flux_matrix))

    relative_weighted_adjoint_error = norm(hodge_weighted_pressure_correction - weighted_divergence_adjoint_correction) /
                                      norm(hodge_weighted_pressure_correction)

    best_gradient_scale = dot(hodge_weighted_pressure_correction, local_covariant_gradient) /
                          dot(local_covariant_gradient, local_covariant_gradient)

    relative_gradient_mismatch = norm(hodge_weighted_pressure_correction - best_gradient_scale * local_covariant_gradient) /
                                 norm(hodge_weighted_pressure_correction)

    @info "OctaHEALPix Hodge projection gradient mismatch" best_gradient_scale relative_gradient_mismatch
    @info "OctaHEALPix Hodge projection exact weighted divergence adjoint" relative_exact_adjoint_error
    @info "OctaHEALPix Hodge-compatible pressure correction" relative_hodge_compatible_pressure_correction_error
    @info "OctaHEALPix Hodge-compatible Schur complement" relative_hodge_compatible_schur_complement_error
    @info "OctaHEALPix Hodge-compatible divergence" relative_hodge_compatible_divergence_error
    @info "OctaHEALPix Hodge-compatible source operator" relative_hodge_compatible_operator_error
    @info "OctaHEALPix Hodge-compatible boundary maps" east_boundary_flux_row_nonzeros north_boundary_flux_row_nonzeros
    @info "OctaHEALPix Hodge-compatible boundary ratio maps" relative_boundary_flux_ratio_error
    @info "OctaHEALPix Hodge projection raw weighted divergence adjoint mismatch" relative_weighted_adjoint_error

    @test all(exact_volume_flux_divergence_row_nonzeros .== 4)
    @test all(east_boundary_flux_row_nonzeros .== 1)
    @test all(north_boundary_flux_row_nonzeros .== 1)
    @test relative_hodge_compatible_divergence_error < 1e-12
    @test relative_hodge_compatible_operator_error < 1e-12
    @test relative_boundary_flux_ratio_error < 1e-12
    @test relative_exact_adjoint_error < 1e-12
    @test relative_hodge_compatible_pressure_correction_error < 1e-12
    @test relative_hodge_compatible_schur_complement_error < 1e-12
    @test dot(pressure, dense_schur_complement) > 0
    @test relative_weighted_adjoint_error > 0.75
    @test relative_gradient_mismatch > 0.75

    set_octahealpix_projection_state!(u, v, initial_state)
    projection = RigidLidProjectionSolver(grid; maxiter = 4Nu, reltol = 1e-12, abstol = 1e-14)
    solve_rigid_lid_projection!(projection, (u = u, v = v), grid)

    solver_correction = octahealpix_projection_face_vector(projection.correction.u, projection.correction.v)
    solver_projected_state = initial_state - solver_correction
    dense_projected_state = hodge_weighted_divergence_free_projection(initial_state, divergence_matrix, hodge_energy_matrix)
    solver_projected_divergence = divergence_matrix * solver_projected_state

    relative_solver_projection_error = norm(solver_projected_state - dense_projected_state) /
                                       norm(dense_projected_state)

    @info "OctaHEALPix rigid-lid projection solve" projection.conjugate_gradient_solver.iteration maximum(abs.(solver_projected_divergence)) relative_solver_projection_error

    @test maximum(abs.(solver_projected_divergence)) < 1e-10
    @test relative_solver_projection_error < 1e-10
end
