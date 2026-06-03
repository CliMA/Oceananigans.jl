using Oceananigans
using Oceananigans.Advection: VectorInvariant, WENOVectorInvariant, U_dot_∇u, U_dot_∇v
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: λnode, Center
using Oceananigans.Operators: Azᶜᶜᵃ, Vᶜᶜᶜ,
                              covariant_kinetic_energyᶜᶜᶜ,
                              covariant_to_volume_flux_uᶠᶜᶜ,
                              covariant_to_volume_flux_vᶜᶠᶜ
using LinearAlgebra: norm
using Test

# This is a no-MPI accuracy gate for OctaHEALPix vector-invariant dynamics.
# It complements the fast smoke/tendency checks in test_spherical_shell_grid.jl
# by requiring Rossby-Haurwitz-style evolution diagnostics to converge against
# an N=16 reference when resolution increases from N=4 to N=8.

function rossby_haurwitz_model(FT, N, momentum_advection)
    radius = one(FT)
    depth = one(FT)
    gravity = one(FT)

    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(N),
                              size = (2N, 2N, 1),
                              z = (-depth, zero(FT)),
                              radius,
                              halo = (5, 5, 3))

    free_surface = ExplicitFreeSurface(gravitational_acceleration = gravity)

    model = HydrostaticFreeSurfaceModel(grid; free_surface,
                                        buoyancy = nothing,
                                        tracers = nothing,
                                        momentum_advection)

    R = radius
    ω = zero(FT)
    K = convert(FT, 1//10)
    n = 4
    Ω = zero(FT)

    function rossby_haurwitz_coefficients(latitude)
        cos_latitude = cos(latitude)
        sin_latitude = sin(latitude)
        cos² = cos_latitude^2
        cos²ⁿ = cos_latitude^(2n)

        A = ω / 2 * (2Ω + ω) * cos² +
            convert(FT, 1//4) * K^2 * ((n + 1) * cos_latitude^(2n + 2) +
                                       (2n^2 - n - 2) * cos²ⁿ -
                                       2n^2 * cos_latitude^(2n - 2))

        B = 2K * (Ω + ω) / ((n + 1) * (n + 2)) *
            cos_latitude^n *
            (n^2 + 2n + 2 - (n + 1)^2 * cos²)

        C = convert(FT, 1//4) * K^2 *
            ((n + 1) * cos_latitude^(2n + 2) - (n + 2) * cos²ⁿ)

        return A, B, C, cos_latitude, sin_latitude
    end

    function uᵢ(λ, φ, z)
        latitude = deg2rad(φ)
        longitude = deg2rad(λ)
        _, _, _, cos_latitude, sin_latitude = rossby_haurwitz_coefficients(latitude)

        return R * ω * cos_latitude +
               R * K * cos_latitude^(n - 1) *
               (n * sin_latitude^2 - cos_latitude^2) *
               cos(n * longitude)
    end

    function vᵢ(λ, φ, z)
        latitude = deg2rad(φ)
        longitude = deg2rad(λ)
        _, _, _, cos_latitude, sin_latitude = rossby_haurwitz_coefficients(latitude)

        return -n * K * R * cos_latitude^(n - 1) *
               sin_latitude *
               sin(n * longitude)
    end

    function ηᵢ(λ, φ, z)
        latitude = deg2rad(φ)
        longitude = deg2rad(λ)
        A, B, C, _, _ = rossby_haurwitz_coefficients(latitude)

        return R^2 / gravity *
               (A + B * cos(n * longitude) + C * cos(2n * longitude))
    end

    η = model.free_surface.displacement

    set!(model, u = uᵢ, v = vᵢ)
    set!(η, ηᵢ)
    fill_halo_regions!(η)

    return model
end

function rossby_haurwitz_surface_diagnostics(model)
    grid = model.grid
    η = model.free_surface.displacement
    k_top = grid.Nz + 1

    area_sum = 0.0
    volume = 0.0
    η² = 0.0
    cos4 = 0.0
    sin4 = 0.0
    cos8 = 0.0
    sin8 = 0.0

    for j in 1:grid.Ny, i in 1:grid.Nx
        area = Azᶜᶜᵃ(i, j, 1, grid)
        λ = deg2rad(λnode(i, j, 1, grid, Center(), Center(), Center()))
        ηij = η[i, j, k_top]

        area_sum += area
        volume += area * ηij
        η² += area * ηij^2
        cos4 += area * ηij * cos(4λ)
        sin4 += area * ηij * sin(4λ)
        cos8 += area * ηij * cos(8λ)
        sin8 += area * ηij * sin(8λ)
    end

    return (volume / area_sum,
            sqrt(η² / area_sum),
            cos4 / area_sum,
            sin4 / area_sum,
            cos8 / area_sum,
            sin8 / area_sum)
end

function short_rossby_haurwitz_diagnostic(FT, N, momentum_advection)
    model = rossby_haurwitz_model(FT, N, momentum_advection)
    Δt = convert(FT, 1//400)

    for _ in 1:2
        time_step!(model, Δt)
    end

    return rossby_haurwitz_surface_diagnostics(model)
end

diagnostic_distance(a, b) = sqrt(sum((a[n] - b[n])^2 for n in eachindex(a)))

function octahealpix_vi_face_counts(grid)
    Nu = (grid.Nx + 1) * grid.Ny
    Nv = grid.Nx * (grid.Ny + 1)
    return Nu, Nv, Nu + Nv
end

octahealpix_vi_u_index(i, j, grid) = (j - 1) * (grid.Nx + 1) + i
octahealpix_vi_v_index(i, j, grid) = (grid.Nx + 1) * grid.Ny + (j - 1) * grid.Nx + i

function set_octahealpix_vi_vector!(u, v, x, grid)
    for j in 1:grid.Ny, i in 1:(grid.Nx + 1)
        u[i, j, 1] = x[octahealpix_vi_u_index(i, j, grid)]
    end

    for j in 1:(grid.Ny + 1), i in 1:grid.Nx
        v[i, j, 1] = x[octahealpix_vi_v_index(i, j, grid)]
    end

    fill_halo_regions!((u, v))

    return nothing
end

function octahealpix_vi_flux_vector(u, v, grid)
    _, _, n = octahealpix_vi_face_counts(grid)
    fluxes = zeros(eltype(grid), n)

    for j in 1:grid.Ny, i in 1:(grid.Nx + 1)
        fluxes[octahealpix_vi_u_index(i, j, grid)] = covariant_to_volume_flux_uᶠᶜᶜ(i, j, 1, grid, u, v)
    end

    for j in 1:(grid.Ny + 1), i in 1:grid.Nx
        fluxes[octahealpix_vi_v_index(i, j, grid)] = covariant_to_volume_flux_vᶜᶠᶜ(i, j, 1, grid, u, v)
    end

    return fluxes
end

function octahealpix_vi_hodge_matrix(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)
    _, _, n = octahealpix_vi_face_counts(grid)
    H = zeros(eltype(grid), n, n)
    x = zeros(eltype(grid), n)

    for column in 1:n
        fill!(x, zero(eltype(grid)))
        x[column] = one(eltype(grid))
        set_octahealpix_vi_vector!(u, v, x, grid)
        H[:, column] .= octahealpix_vi_flux_vector(u, v, grid)
    end

    return H
end

function octahealpix_divergence_free_flux_vector(grid)
    FT = eltype(grid)
    Nx, Ny = grid.Nx, grid.Ny
    ψ = zeros(FT, Nx + 1, Ny + 1)

    for j in 1:(Ny + 1), i in 1:(Nx + 1)
        x = 2π * (i - 1) / Nx
        y = π * (j - 1) / Ny
        ψ[i, j] = sin(x) * (convert(FT, 1//4) + sin(y)^2)
    end

    _, _, n = octahealpix_vi_face_counts(grid)
    fluxes = zeros(FT, n)

    for j in 1:Ny, i in 1:(Nx + 1)
        fluxes[octahealpix_vi_u_index(i, j, grid)] = ψ[i, j + 1] - ψ[i, j]
    end

    for j in 1:(Ny + 1), i in 1:Nx
        fluxes[octahealpix_vi_v_index(i, j, grid)] = -(ψ[i + 1, j] - ψ[i, j])
    end

    return fluxes
end

function octahealpix_vi_max_flux_divergence(fluxes, grid)
    FT = eltype(grid)
    maximum_divergence = zero(FT)

    for j in 1:grid.Ny, i in 1:grid.Nx
        divergence = fluxes[octahealpix_vi_u_index(i + 1, j, grid)] - fluxes[octahealpix_vi_u_index(i, j, grid)] +
                     fluxes[octahealpix_vi_v_index(i, j + 1, grid)] - fluxes[octahealpix_vi_v_index(i, j, grid)]
        maximum_divergence = max(maximum_divergence, abs(divergence))
    end

    return maximum_divergence
end

function octahealpix_vi_kinetic_energy(model)
    grid = model.grid
    u = model.velocities.u
    v = model.velocities.v
    energy = zero(eltype(grid))

    for j in 1:grid.Ny, i in 1:grid.Nx
        energy += Vᶜᶜᶜ(i, j, 1, grid) * covariant_kinetic_energyᶜᶜᶜ(i, j, 1, grid, u, v)
    end

    return energy
end

function shift_octahealpix_vi_velocity!(model, Gu, Gv, factor)
    grid = model.grid
    u = model.velocities.u
    v = model.velocities.v

    for j in 1:grid.Ny, i in 1:(grid.Nx + 1)
        u[i, j, 1] += factor * Gu[i, j, 1]
    end

    for j in 1:(grid.Ny + 1), i in 1:grid.Nx
        v[i, j, 1] += factor * Gv[i, j, 1]
    end

    fill_halo_regions!(model.velocities)

    return nothing
end

function octahealpix_vi_energy_directional_derivative(model, Gu, Gv)
    ϵ = convert(eltype(model.grid), 1//1000000)
    initial_energy = octahealpix_vi_kinetic_energy(model)

    shift_octahealpix_vi_velocity!(model, Gu, Gv, +ϵ)
    energy_plus = octahealpix_vi_kinetic_energy(model)
    shift_octahealpix_vi_velocity!(model, Gu, Gv, -2ϵ)
    energy_minus = octahealpix_vi_kinetic_energy(model)
    shift_octahealpix_vi_velocity!(model, Gu, Gv, +ϵ)

    return (energy_plus - energy_minus) / (2ϵ), initial_energy
end

function octahealpix_vi_tendency_fields(model)
    grid = model.grid
    scheme = model.advection.momentum
    velocities = model.velocities
    Gu = XFaceField(grid)
    Gv = YFaceField(grid)

    for j in 1:grid.Ny, i in 1:(grid.Nx + 1)
        Gu[i, j, 1] = -U_dot_∇u(i, j, 1, grid, scheme, velocities)
    end

    for j in 1:(grid.Ny + 1), i in 1:grid.Nx
        Gv[i, j, 1] = -U_dot_∇v(i, j, 1, grid, scheme, velocities)
    end

    fill_halo_regions!((Gu, Gv))

    return Gu, Gv
end

function test_octahealpix_centered_vi_energy_conservation_for_divergence_free_transport(FT, N)
    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(N),
                              size = (2N, 2N, 1),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              halo = (5, 5, 3))

    H = octahealpix_vi_hodge_matrix(grid)
    target_fluxes = octahealpix_divergence_free_flux_vector(grid)

    @test octahealpix_vi_max_flux_divergence(target_fluxes, grid) < 100eps(FT)

    covariant_velocity_vector = H \ target_fluxes
    u = XFaceField(grid)
    v = YFaceField(grid)
    set_octahealpix_vi_vector!(u, v, covariant_velocity_vector, grid)
    reconstructed_fluxes = octahealpix_vi_flux_vector(u, v, grid)

    @test norm(reconstructed_fluxes - target_fluxes) / norm(target_fluxes) < 100eps(FT)
    @test octahealpix_vi_max_flux_divergence(reconstructed_fluxes, grid) < 100eps(FT)

    model = HydrostaticFreeSurfaceModel(grid; tracers = (),
                                        buoyancy = nothing,
                                        coriolis = nothing,
                                        free_surface = nothing,
                                        closure = nothing,
                                        momentum_advection = VectorInvariant(FT))

    set_octahealpix_vi_vector!(model.velocities.u, model.velocities.v, covariant_velocity_vector, grid)

    Gu, Gv = octahealpix_vi_tendency_fields(model)
    energy_tendency, initial_energy = octahealpix_vi_energy_directional_derivative(model, Gu, Gv)

    @test abs(energy_tendency) / initial_energy < sqrt(eps(FT))

    return nothing
end

function test_rossby_haurwitz_vector_invariant_reference_convergence(momentum_advection)
    FT = Float64

    coarse_diagnostic = short_rossby_haurwitz_diagnostic(FT, 4, momentum_advection)
    fine_diagnostic = short_rossby_haurwitz_diagnostic(FT, 8, momentum_advection)
    reference_diagnostic = short_rossby_haurwitz_diagnostic(FT, 16, momentum_advection)

    coarse_error = diagnostic_distance(coarse_diagnostic, reference_diagnostic)
    fine_error = diagnostic_distance(fine_diagnostic, reference_diagnostic)

    @info "Rossby-Haurwitz vector-invariant diagnostic convergence" summary(momentum_advection) coarse_error fine_error ratio = coarse_error / fine_error

    @test all(isfinite, reference_diagnostic)
    @test fine_error < coarse_error

    return nothing
end

@testset "OctaHEALPix vector-invariant energy conservation" begin
    FT = Float64

    test_octahealpix_centered_vi_energy_conservation_for_divergence_free_transport(FT, 4)
    test_octahealpix_centered_vi_energy_conservation_for_divergence_free_transport(FT, 8)
end

@testset "OctaHEALPix vector-invariant Rossby-Haurwitz accuracy" begin
    FT = Float64

    test_rossby_haurwitz_vector_invariant_reference_convergence(VectorInvariant(FT))
    test_rossby_haurwitz_vector_invariant_reference_convergence(WENOVectorInvariant(FT; order = 3))
    test_rossby_haurwitz_vector_invariant_reference_convergence(WENOVectorInvariant(FT; order = 5))
end
