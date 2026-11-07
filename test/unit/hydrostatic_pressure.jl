include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!
using Oceananigans.Operators: Δzᶜᶜᶜ

function hydrostatic_pressure_test_grid(arch, FT, discretization)
    z_faces = [-100, -60, -30, -10, 0]
    z = discretization == :uniform ? (-100, 0) :
        discretization == :stretched ? z_faces : MutableVerticalDiscretization(z_faces)
    return RectilinearGrid(arch, FT; size=(2, 2, 4), x=(0, 1), y=(0, 1), z, topology=(Periodic, Periodic, Bounded))
end

function uniform_buoyancy_hydrostatic_pressure(model, grid, b₀)
    set!(model, b=b₀)
    pressures = model isa NonhydrostaticModel ? model.pressures : model.pressure
    update_hydrostatic_pressure!(pressures.pHY′, architecture(grid), grid, model.buoyancy, model.tracers)
    return Array(interior(pressures.pHY′))
end

function energy_drift_and_top_source(arch, FT)
    Nx, Nz = 32, 8
    g = FT(1//10)
    Δt = 20
    Nt = 200
    grid = RectilinearGrid(arch, FT; size=(Nx, Nz), x=(0, 1e6), z=(-1000, 0), topology=(Periodic, Flat, Bounded))

    model = HydrostaticFreeSurfaceModel(grid; buoyancy=BuoyancyTracer(), tracers=:b,
                                        free_surface = ExplicitFreeSurface(gravitational_acceleration=g),
                                        momentum_advection = nothing,
                                        tracer_advection = Centered(order=2),
                                        vertical_coordinate = ZCoordinate(),
                                        timestepper = :SplitRungeKutta3)

    set!(model, b=(x, z) -> 1e-5 * z + 1e-2 * sin(2π * x / 1e6) * exp(z / 200))

    Δz = Δzᶜᶜᶜ(1, 1, 1, grid)
    z = znodes(grid, Center())

    function total_energy()
        u = Array(interior(model.velocities.u))
        b = Array(interior(model.tracers.b))
        η = Array(interior(model.free_surface.displacement))
        kinetic = sum(u.^2) / 2 * Δz
        potential = - sum(b[i, 1, k] * z[k] for i in 1:Nx, k in 1:Nz) * Δz
        surface = g * sum(η.^2) / 2
        return kinetic + potential + surface
    end

    E₀ = total_energy()
    top_source = zero(FT)

    for n in 1:Nt
        w = Array(interior(model.velocities.w))
        b = Array(interior(model.tracers.b))
        top_source += abs(sum(b[i, 1, Nz] * w[i, 1, Nz+1] for i in 1:Nx)) * Δz / 2 * Δt
        time_step!(model, Δt)
    end

    return abs(total_energy() - E₀), top_source
end

@testset "Hydrostatic pressure anomaly" begin
    @info "Testing the hydrostatic pressure anomaly..."

    for arch in archs, FT in float_types
        @testset "Uniform buoyancy gives p = b z at cell centers [$FT, $(typeof(arch))]" begin
            b₀ = FT(2)
            for discretization in (:uniform, :stretched, :mutable)
                grid = hydrostatic_pressure_test_grid(arch, FT, discretization)
                z = znodes(grid, Center())

                hydrostatic_model = HydrostaticFreeSurfaceModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
                p = uniform_buoyancy_hydrostatic_pressure(hydrostatic_model, grid, b₀)
                @test all(p[1, 1, k] ≈ b₀ * z[k] for k in 1:grid.Nz)

                if discretization != :mutable
                    nonhydrostatic_model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b,
                                                               hydrostatic_pressure_anomaly=CenterField(grid))
                    p = uniform_buoyancy_hydrostatic_pressure(nonhydrostatic_model, grid, b₀)
                    @test all(p[1, 1, k] ≈ b₀ * z[k] for k in 1:grid.Nz)
                end
            end
        end

        @testset "Top value does not depend on the halo [$FT, $(typeof(arch))]" begin
            grid = hydrostatic_pressure_test_grid(arch, FT, :uniform)
            b₀ = FT(2)
            b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(3b₀))
            model = HydrostaticFreeSurfaceModel(grid; buoyancy=BuoyancyTracer(), tracers=:b,
                                                boundary_conditions=(; b=b_bcs))
            p = uniform_buoyancy_hydrostatic_pressure(model, grid, b₀)
            @test p[1, 1, grid.Nz] ≈ b₀ * znodes(grid, Center())[grid.Nz]
        end
    end

    @testset "Pressure work closes the energy budget with a free surface" begin
        drift, top_source = energy_drift_and_top_source(CPU(), Float64)
        @test top_source > 0
        @test drift < 1e-4 * top_source
    end
end
