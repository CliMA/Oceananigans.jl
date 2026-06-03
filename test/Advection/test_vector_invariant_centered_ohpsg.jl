using Test
using Oceananigans
using Oceananigans.Advection: VectorInvariant
using Oceananigans.Operators: horizontal_volume_flux_div_xyᶜᶜᶜ
using Random

@inline function ohpsg_random_vortex_unit_dot(λ, φ, λ0, φ0)
    return cosd(φ) * cosd(φ0) * cosd(λ - λ0) + sind(φ) * sind(φ0)
end

function ohpsg_random_vortex_streamfunction(seed)
    Random.seed!(seed)

    n_vortices = 24
    σ_vortex = 0.35
    λ_vortices = [180 * (2rand() - 1) for _ in 1:n_vortices]
    φ_vortices = [50 * (2rand() - 1) for _ in 1:n_vortices]
    amplitudes = [0.3 * (2rand() - 1) for _ in 1:n_vortices]

    function ψ(λ, φ)
        value = 0.0

        for n in 1:n_vortices
            q = clamp(ohpsg_random_vortex_unit_dot(λ, φ, λ_vortices[n], φ_vortices[n]), -1, 1)
            distance = acos(q)
            value += amplitudes[n] * exp(-(distance / σ_vortex)^2 / 2)
        end

        return value
    end

    return ψ
end

function ohpsg_centered_vi_random_vortex_model(; N = 32, seed = 42)
    FT = Float64
    grid = SphericalShellGrid(CPU(), FT;
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

    ψ = ohpsg_random_vortex_streamfunction(seed)

    function u_init(λ, φ, z)
        h = 1e-4
        return (ψ(λ, φ + h) - ψ(λ, φ - h)) / (2h) * 180 / π
    end

    function v_init(λ, φ, z)
        h = 1e-4
        cos_φ = max(cosd(φ), 0.01)
        return -(ψ(λ + h, φ) - ψ(λ - h, φ)) / (2h) * 180 / π / cos_φ
    end

    set!(model, u = u_init, v = v_init)

    return model
end

function ohpsg_horizontal_velocity_diagnostics(model)
    grid = model.grid
    u = model.velocities.u
    v = model.velocities.v
    projection = model.rigid_lid_projection

    maximum_u = zero(eltype(grid))
    maximum_v = zero(eltype(grid))
    seam_maximum_u = zero(eltype(grid))
    seam_maximum_v = zero(eltype(grid))
    maximum_horizontal_divergence = zero(eltype(grid))
    maximum_u_location = (0, 0)
    maximum_v_location = (0, 0)

    for j in 1:grid.Ny, i in 1:grid.Nx
        absolute_u = abs(u[i, j, 1])
        absolute_v = abs(v[i, j, 1])
        divergence = horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid, u, v)
        seam = i <= 2 || i >= grid.Nx - 1 || j <= 2 || j >= grid.Ny - 1

        seam_maximum_u = ifelse(seam, max(seam_maximum_u, absolute_u), seam_maximum_u)
        seam_maximum_v = ifelse(seam, max(seam_maximum_v, absolute_v), seam_maximum_v)
        maximum_horizontal_divergence = max(maximum_horizontal_divergence, abs(divergence))

        if absolute_u > maximum_u
            maximum_u = absolute_u
            maximum_u_location = (i, j)
        end

        if absolute_v > maximum_v
            maximum_v = absolute_v
            maximum_v_location = (i, j)
        end
    end

    projection_iterations = isnothing(projection) ? 0 : projection.conjugate_gradient_solver.iteration

    return (; maximum_u, maximum_v,
              seam_maximum_u, seam_maximum_v,
              maximum_u_location, maximum_v_location,
              maximum_horizontal_divergence,
              projection_iterations)
end

function run_ohpsg_centered_vi_random_vortex_gate(; N = 32, seed = 42,
                                                  Δt = 9.39248163e-03,
                                                  steps = 533,
                                                  progress_interval = 0)
    model = ohpsg_centered_vi_random_vortex_model(; N, seed)

    for step in 1:steps
        time_step!(model, Δt)
        diagnostics = ohpsg_horizontal_velocity_diagnostics(model)

        if progress_interval > 0 && step % progress_interval == 0
            @info "Centered VI OHPSG random-vortex progress" N step model.clock.time diagnostics.maximum_u diagnostics.maximum_v diagnostics.seam_maximum_u diagnostics.seam_maximum_v diagnostics.maximum_horizontal_divergence diagnostics.projection_iterations
        end

        if !isfinite(diagnostics.maximum_u) || !isfinite(diagnostics.maximum_v)
            return (; passed = false, step, time = model.clock.time, diagnostics...)
        end

        if max(diagnostics.maximum_u, diagnostics.maximum_v) > 1e3
            return (; passed = false, step, time = model.clock.time, diagnostics...)
        end
    end

    diagnostics = ohpsg_horizontal_velocity_diagnostics(model)

    return (; passed = true,
              step = steps,
              time = model.clock.time,
              diagnostics...)
end

@testset "Centered VectorInvariant projected random-vortex OHPSG short gate" begin
    result = run_ohpsg_centered_vi_random_vortex_gate(; N = 16,
                                                       steps = 180,
                                                       progress_interval = 20)

    @info "Centered VI OHPSG short random-vortex gate" result.step result.time result.maximum_u result.maximum_v result.seam_maximum_u result.seam_maximum_v result.maximum_horizontal_divergence result.projection_iterations result.maximum_u_location result.maximum_v_location

    @test result.passed
    @test result.step == 180
    @test result.time ≈ 1.6906466934 atol=1e-10
    @test result.maximum_u < 0.2
    @test result.maximum_v < 0.2
    @test result.maximum_horizontal_divergence < 1e-10
end

@testset "Centered VectorInvariant random-vortex OHPSG extended gate" begin
    if get(ENV, "EXTENDED_OHPSG_VI_TESTS", "false") == "true"
        result = run_ohpsg_centered_vi_random_vortex_gate(; progress_interval = 50)

        @info "Centered VI OHPSG extended random-vortex gate" result.step result.time result.maximum_u result.maximum_v result.seam_maximum_u result.seam_maximum_v result.maximum_horizontal_divergence result.projection_iterations result.maximum_u_location result.maximum_v_location

        @test result.passed
        @test result.step == 533
        @test result.time ≈ 5.00619310879 atol=1e-10
        @test result.maximum_u < 0.2
        @test result.maximum_v < 0.2
        @test result.maximum_horizontal_divergence < 1e-10
    else
        @test_skip false
    end
end
