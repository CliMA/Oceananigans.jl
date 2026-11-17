include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans
using Oceananigans: PrescribedVelocityFields

@inline unit_source(x, y, z, t) = 1

# An ideal age tracer: it ages at one second per second and is reset to zero in the top layer.
function ideal_age_column(arch, timestepper, time_discretization; Δt=1, rate=1000/Δt, steps=200)
    grid = RectilinearGrid(arch; size=(1, 1, 8), extent=(1, 1, 8), topology=(Periodic, Periodic, Bounded))
    top_layer = CenterField(grid)
    interior(top_layer, :, :, grid.Nz) .= 1
    reset = Relaxation(; rate, mask=top_layer, time_discretization)

    model = HydrostaticFreeSurfaceModel(grid; timestepper, tracers=:age, buoyancy=nothing,
                                        velocities=PrescribedVelocityFields(),
                                        forcing=(; age=(unit_source, reset)))
    for _ in 1:steps
        time_step!(model, Δt)
    end

    return Array(interior(model.tracers.age, 1, 1, :)), model.clock.time
end

function relaxed_column(arch, grid, time_discretization; Δt, rate, target, c₀)
    forcing = (; c=Relaxation(; rate, target, time_discretization))
    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields(), forcing)
    set!(model, c=c₀)
    time_step!(model, Δt)
    return Array(interior(model.tracers.c, 1, 1, :))
end

@testset "Implicit Relaxation" begin
    @test_throws TypeError Relaxation(rate=1, time_discretization=VerticallyImplicitTimeDiscretization())

    for arch in archs
        @testset "Ideal age reset in the top layer, rate Δt = 1000 [$(typeof(arch)), $timestepper]" for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
            rate = 1000
            explicit_age, _ = ideal_age_column(arch, timestepper, ExplicitTimeDiscretization(); rate)
            @test !all(isfinite, explicit_age)

            age, time = ideal_age_column(arch, timestepper, ImplicitTimeDiscretization(); rate)
            @test all(age[1:end-1] .≈ time)
            @test age[end] ≈ 1 / rate
        end

        @testset "One backward Euler step toward a target [$(typeof(arch))]" begin
            Δt, rate, target, c₀ = 1, 100, 2, 1
            column = RectilinearGrid(arch; size=4, z=(-4, 0), topology=(Flat, Flat, Bounded))
            immersed_column = ImmersedBoundaryGrid(column, GridFittedBottom(-2))
            c★ = (c₀ + Δt * rate * target) / (1 + Δt * rate)

            c = relaxed_column(arch, column, ImplicitTimeDiscretization(); Δt, rate, target, c₀)
            @test all(c .≈ c★)

            c = relaxed_column(arch, immersed_column, ImplicitTimeDiscretization(); Δt, rate, target, c₀)
            @test all(c[3:4] .≈ c★)

            c = relaxed_column(arch, column, ExplicitTimeDiscretization(); Δt, rate, target, c₀)
            @test all(abs.(c) .> abs(target))
        end

        @testset "Implicit velocity sponge in NonhydrostaticModel [$(typeof(arch))]" begin
            grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1))
            sponge = Relaxation(rate=1000, time_discretization=ImplicitTimeDiscretization())
            model = NonhydrostaticModel(grid; forcing=(; u=sponge))
            set!(model, u=1)
            time_step!(model, 1)
            u = Array(interior(model.velocities.u))
            @test all(0 .< u .< 1e-2)
        end

        @testset "Implicit relaxation of turbulent tracers [$(typeof(arch)), $(nameof(typeof(closure)))]" for (closure, names) in
                ((CATKEVerticalDiffusivity(), (:e,)), (TKEDissipationVerticalDiffusivity(), (:e, :ϵ)))
            grid = RectilinearGrid(arch; size=8, z=(-8, 0), topology=(Flat, Flat, Bounded))
            target = 1e-3
            relaxation = Relaxation(; rate=1000, target, time_discretization=ImplicitTimeDiscretization())
            model = HydrostaticFreeSurfaceModel(grid; closure, buoyancy=BuoyancyTracer(), tracers=:b,
                                                forcing=NamedTuple{names}(ntuple(Returns(relaxation), length(names))))
            set!(model, b = z -> 1e-5 * z)
            for _ in 1:10
                time_step!(model, 1)
            end
            for name in names
                @test all(isapprox.(Array(interior(model.tracers[name])), target, rtol=1e-2))
            end
        end
    end
end
