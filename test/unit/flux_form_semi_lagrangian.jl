include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "volume_integrals.jl"))

using Oceananigans.Advection: MonotonePPMLimiter, maximum_courant_number
using Oceananigans.Grids: required_halo_size_x, required_halo_size_y, required_halo_size_z

#####
##### Helpers
#####

function one_dimensional_ffsl_model(arch, N, u; maximum_courant_number = 3, limiter = :monotone)
    H = maximum_courant_number + 3
    grid = RectilinearGrid(arch; size = (N, 1), halo = (H, 3), x = (0, N), z = (-1, 0),
                           topology = (Periodic, Flat, Bounded))
    tracer_advection = FluxFormSemiLagrangian(; maximum_courant_number, limiter)
    return HydrostaticFreeSurfaceModel(grid; velocities = PrescribedVelocityFields(; u), tracers = :c,
                                       tracer_advection, timestepper = :SplitRungeKutta3, buoyancy = nothing)
end

# Discretely non-divergent velocities from a streamfunction ψ(x, y, t) evaluated at cell corners
function streamfunction_velocities(ψ, Δ)
    u(x, y, z, t) = - (ψ(x, y + Δ/2, t) - ψ(x, y - Δ/2, t)) / Δ
    v(x, y, z, t) = + (ψ(x + Δ/2, y, t) - ψ(x - Δ/2, y, t)) / Δ
    return PrescribedVelocityFields(; u, v)
end

function two_dimensional_ffsl_model(arch, N, ψ, topology)
    grid = RectilinearGrid(arch; size = (N, N, 1), halo = (6, 6, 3), x = (0, 1), y = (0, 1), z = (-1, 0), topology)
    velocities = streamfunction_velocities(ψ, 1 / N)
    return HydrostaticFreeSurfaceModel(grid; velocities, tracers = (:c, :uniform),
                                       tracer_advection = FluxFormSemiLagrangian(),
                                       timestepper = :SplitRungeKutta3, buoyancy = nothing)
end

line(c) = Array(interior(c))[:, 1, 1]
plane(c) = Array(interior(c))[:, :, 1]

function time_step_n!(model, Δt, Nsteps)
    for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return nothing
end

# Periodic analogue of the deformational flow of Lauritzen et al. (2012, case 4); it reverses at t = 1/2
deformational_ψ(x, y, t) = 1 / π * sin(π * (x - t))^2 * sin(π * y)^2 * cos(π * t) - y

# Solid-body rotation in a disk of radius 0.45 centred in a closed box, zero velocity outside
disk_rotation_ψ(x, y, t) = π * min((x - 1/2)^2 + (y - 1/2)^2, 0.45^2)

cosine_bell(x, y) = max(0, (1 + cos(π * min(1, sqrt((x - 0.35)^2 + (y - 0.5)^2) / 0.15))) / 2) + 0.1

#####
##### Tests
#####

@testset "FluxFormSemiLagrangian construction and validation" begin
    @info "Testing FluxFormSemiLagrangian construction and validation..."

    scheme = FluxFormSemiLagrangian()
    @test maximum_courant_number(scheme) == 3
    @test scheme.limiter isa MonotonePPMLimiter
    @test scheme.vertical_scheme isa WENO
    @test required_halo_size_x(scheme) == 6
    @test required_halo_size_y(scheme) == 6
    @test required_halo_size_z(scheme) == 3

    unlimited = FluxFormSemiLagrangian(; maximum_courant_number = 2, limiter = nothing, vertical_scheme = Centered())
    @test maximum_courant_number(unlimited) == 2
    @test isnothing(unlimited.limiter)
    @test required_halo_size_x(unlimited) == 5
    @test required_halo_size_z(unlimited) == 1

    @test_throws ArgumentError FluxFormSemiLagrangian(; maximum_courant_number = 0)
    @test_throws ArgumentError FluxFormSemiLagrangian(; maximum_courant_number = 2.5)
    @test_throws ArgumentError FluxFormSemiLagrangian(; limiter = :positive)

    @test occursin("FluxFormSemiLagrangian", summary(scheme))

    for arch in archs
        small_halo_grid = RectilinearGrid(arch; size = (8, 8, 2), halo = (3, 3, 3), extent = (1, 1, 1))
        @test_throws ArgumentError HydrostaticFreeSurfaceModel(small_halo_grid; tracers = :c,
                                                               tracer_advection = FluxFormSemiLagrangian(),
                                                               timestepper = :SplitRungeKutta3)

        grid = RectilinearGrid(arch; size = (8, 8, 4), halo = (6, 6, 3), extent = (1, 1, 1))
        @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers = :c,
                                                               tracer_advection = FluxFormSemiLagrangian(),
                                                               timestepper = :QuasiAdamsBashforth2)

        @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers = (:c, :d),
                                                               tracer_advection = (c = FluxFormSemiLagrangian(),
                                                                                   d = FluxFormSemiLagrangian(maximum_courant_number=2)),
                                                               timestepper = :SplitRungeKutta3)

        @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers = :c, closure = CATKEVerticalDiffusivity(),
                                                               buoyancy = nothing,
                                                               tracer_advection = FluxFormSemiLagrangian(),
                                                               timestepper = :SplitRungeKutta3)

        @test_throws ArgumentError NonhydrostaticModel(grid; tracers = :c, tracer_advection = FluxFormSemiLagrangian())

        model = HydrostaticFreeSurfaceModel(grid; tracers = (:c, :d),
                                            tracer_advection = (c = FluxFormSemiLagrangian(), d = WENO()),
                                            timestepper = :SplitRungeKutta3)
        @test model.advection.c isa FluxFormSemiLagrangian
        @test model.advection.d isa WENO
        @test model.advection.c.workspace.qˣ isa Field
    end
end

@testset "One-dimensional FluxFormSemiLagrangian advection" begin
    for arch in archs
        @info "Testing one-dimensional FluxFormSemiLagrangian advection on $arch..."
        N = 32
        square_wave(x, z) = N/4 < x < N/2 ? 1 : 0

        @testset "Conservation, uniform tracer and bounds at C = $C" for C in (0.5, 1.5, 2.7)
            model = one_dimensional_ffsl_model(arch, N, (x, z, t) -> C)
            set!(model, c = (x, z) -> 1 + sin(2π * x / N) / 2 + rand() / 10)
            ∫c₀ = volume_integral(model.tracers.c)
            time_step_n!(model, 1, 20)
            ∫c₁ = volume_integral(model.tracers.c)
            @test abs(∫c₁ - ∫c₀) / ∫c₀ < 1e-14

            set!(model, c = 1)
            time_step_n!(model, 1, 20)
            @test maximum(abs, line(model.tracers.c) .- 1) < 1e-14

            set!(model, c = square_wave)
            time_step_n!(model, 1, 20)
            c = line(model.tracers.c)
            @test minimum(c) > - 1e-14
            @test maximum(c) < 1 + 1e-14
        end

        @testset "Exact shift at integer Courant number C = $C" for C in (1, 2, 3)
            model = one_dimensional_ffsl_model(arch, N, (x, z, t) -> C)
            set!(model, c = (x, z) -> rand())
            c₀ = line(model.tracers.c)
            time_step!(model, 1)
            @test maximum(abs, line(model.tracers.c) .- circshift(c₀, C)) < 1e-14
        end

        @testset "Variable Courant number crossing integers" begin
            u(x, z, t) = 1.8 + 1.2 * sin(2π * x / N)
            model = one_dimensional_ffsl_model(arch, N, u)
            set!(model, c = (x, z) -> 1 / u(x, z, 0))
            ∫c₀ = volume_integral(model.tracers.c)
            time_step_n!(model, 1, N)
            c₁ = line(model.tracers.c)
            @test abs(volume_integral(model.tracers.c) - ∫c₀) / ∫c₀ < 1e-14
            @test all(isfinite, c₁)
            @test minimum(c₁) > 0
        end

        @testset "Convergence on a smooth profile at C = 1.5 (limiter = $limiter)" for (limiter, expected_order) in ((nothing, 2.8), (:monotone, 2))
            errors = map((48, 96, 192)) do N
                model = one_dimensional_ffsl_model(arch, N, (x, z, t) -> 3/2; limiter)
                set!(model, c = (x, z) -> sin(2π * x / N))
                c₀ = line(model.tracers.c)
                time_step_n!(model, 1, 2N ÷ 3)
                sum(abs, line(model.tracers.c) .- c₀) / N
            end
            orders = log2.(errors[1:end-1] ./ errors[2:end])
            @info "    L1 errors $errors, convergence orders $orders (limiter = $limiter)"
            @test all(orders .> expected_order)
        end
    end
end

@testset "Two-dimensional FluxFormSemiLagrangian advection at Courant numbers above 1" begin
    for arch in archs
        @info "Testing two-dimensional FluxFormSemiLagrangian advection on $arch..."
        N = 32

        for (flow, ψ, topology, Nsteps) in (("deformational flow", deformational_ψ, (Periodic, Periodic, Bounded), 24),
                                            ("solid-body rotation", disk_rotation_ψ, (Bounded, Bounded, Bounded), 32))
            @testset "Uniform tracer and conservation in $flow" begin
                model = two_dimensional_ffsl_model(arch, N, ψ, topology)
                set!(model, c = (x, y, z) -> cosine_bell(x, y) + rand() / 10, uniform = 1)

                Δt = 1 / Nsteps
                u, v, _ = model.velocities
                C = Δt * N * max(maximum(abs, interior(Field(u))), maximum(abs, interior(Field(v))))
                @test C > 2

                ∫c₀ = volume_integral(model.tracers.c)
                maximum_uniform_error = 0.0
                for _ in 1:Nsteps
                    time_step!(model, Δt)
                    maximum_uniform_error = max(maximum_uniform_error, maximum(abs, plane(model.tracers.uniform) .- 1))
                end
                c₁ = plane(model.tracers.c)
                mass_error = abs(volume_integral(model.tracers.c) - ∫c₀) / ∫c₀

                @info "    $flow at C = $C: uniform tracer error $maximum_uniform_error, relative mass error $mass_error"

                @test maximum_uniform_error < 1e-13
                @test mass_error < 1e-13
                @test all(isfinite, c₁)
            end
        end
    end
end
