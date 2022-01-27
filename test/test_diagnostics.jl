using Oceananigans.Fields: FieldSlicer
using Oceananigans.Diagnostics
using Oceananigans.Diagnostics: AbstractDiagnostic

struct TestDiagnostic <: AbstractDiagnostic end

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant

TestModel_VerticallyStrectedRectGrid(arch, FT, ν=1.0, Δx=0.5) =
    NonhydrostaticModel(
          grid = RectilinearGrid(arch, FT, size=(3, 3, 3), x=(0, 3Δx), y=(0, 3Δx), z=0:Δx:3Δx,),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν)
)


TestModel_RegularRectGrid(arch, FT, ν=1.0, Δx=0.5) =
    NonhydrostaticModel(
          grid = RectilinearGrid(arch, FT, topology=(Periodic, Periodic, Periodic), size=(3, 3, 3), extent=(3Δx, 3Δx, 3Δx)),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν)
)

function diffusive_cfl_diagnostic_is_correct(arch, FT)
    Δt = FT(1.3e-6)
    Δx = FT(0.5)
    ν = FT(1.2)
    CFL_by_hand = Δt * ν / Δx^2

    model = TestModel_RegularRectGrid(arch, FT, ν, Δx)
    cfl = DiffusiveCFL(FT(Δt))

    return cfl(model) ≈ CFL_by_hand
end

function advective_cfl_diagnostic_is_correct_on_regular_grid(arch, FT)
    model = TestModel_RegularRectGrid(arch, FT)

    Δt = FT(1.3e-6)
    Δx = FT(model.grid.Δxᶜᵃᵃ)
    u₀ = FT(1.2)
    CFL_by_hand = Δt * u₀ / Δx

    set!(model, u=u₀)
    cfl = AdvectiveCFL(FT(Δt))

    return cfl(model) ≈ CFL_by_hand
end

function advective_cfl_diagnostic_is_correct_on_vertically_stretched_grid(arch, FT)
    model = TestModel_VerticallyStrectedRectGrid(arch, FT)

    Δt = FT(1.3e-6)
    Δx = FT(model.grid.Δxᶜᵃᵃ)
    u₀ = FT(1.2)
    CFL_by_hand = Δt * u₀ / Δx

    set!(model, u=u₀)
    cfl = AdvectiveCFL(FT(Δt))

    return cfl(model) ≈ CFL_by_hand
end

function accurate_advective_cfl_on_regular_grid(arch, FT)
    model = TestModel_RegularRectGrid(arch, FT)

    Δt = FT(1.7)

    Δx = model.grid.Δxᶜᵃᵃ
    Δy = model.grid.Δyᵃᶜᵃ
    Δz = model.grid.Δzᵃᵃᶜ

    u₀ = FT(1.2)
    v₀ = FT(-2.5)
    w₀ = FT(3.9)

    set!(model, u=u₀, v=v₀, w=w₀)

    CFL_by_hand = Δt * (abs(u₀) / Δx + abs(v₀) / Δy + abs(w₀) / Δz)

    cfl = CFL(FT(Δt), Oceananigans.Diagnostics.accurate_cell_advection_timescale)

    return cfl(model) ≈ CFL_by_hand
end

function accurate_advective_cfl_on_stretched_grid(arch, FT)
    grid = RectilinearGrid(arch, size=(4, 4, 8), x=(0, 100), y=(0, 100), z=[k^2 for k in 0:8])
    model = NonhydrostaticModel(grid=grid)

    Δt = FT(15.5)

    Δx = model.grid.Δxᶜᵃᵃ
    Δy = model.grid.Δyᵃᶜᵃ

    # At k = 1, w = 0 so the CFL constraint happens at the second face (k = 2).
    Δz_min = CUDA.@allowscalar Oceananigans.Operators.Δzᵃᵃᶠ(1, 1, 2, grid)

    u₀ = FT(1.2)
    v₀ = FT(-2.5)
    w₀ = FT(3.9)

    set!(model, u=u₀, v=v₀, w=w₀, enforce_incompressibility=false)

    CFL_by_hand = Δt * (abs(u₀) / Δx + abs(v₀) / Δy + abs(w₀) / Δz_min)

    cfl = CFL(FT(Δt), Oceananigans.Diagnostics.accurate_cell_advection_timescale)

    return cfl(model) ≈ CFL_by_hand
end

function accurate_advective_cfl_on_lat_lon_grid(arch, FT)
    grid = LatitudeLongitudeGrid(arch, size=(8, 8, 8), longitude=(-10, 10), latitude=(0, 45), z=(-1000, 0))
    model = HydrostaticFreeSurfaceModel(grid=grid, momentum_advection=VectorInvariant())

    Δt = FT(1000)

    Nx, Ny, Nz = size(grid)

    # Will be the smallest at higher latitudes.
    Δx_min = CUDA.@allowscalar Oceananigans.Operators.Δxᶠᶜᵃ(1, Ny, 1, grid)

    # Will be the same at every grid point.
    Δy_min = CUDA.@allowscalar Oceananigans.Operators.Δyᶜᶠᵃ(1, 1, 1, grid)

    Δz = model.grid.Δzᵃᵃᶠ

    u₀ = FT(1.2)
    v₀ = FT(-2.5)
    w₀ = FT(-0.1)

    # Avoid recomputation of w
    set!(model.velocities.u, u₀)
    set!(model.velocities.v, v₀)
    set!(model.velocities.w, w₀)

    CFL_by_hand = Δt * (abs(u₀) / Δx_min + abs(v₀) / Δy_min + abs(w₀) / Δz)

    cfl = CFL(FT(Δt), Oceananigans.Diagnostics.accurate_cell_advection_timescale)

    return cfl(model) ≈ CFL_by_hand
end

get_iteration(model) = model.clock.iteration
get_time(model) = model.clock.time

function diagnostics_getindex(arch, FT)
    model = TestModel_RegularRectGrid(arch, FT)
    simulation = Simulation(model, Δt=0, stop_iteration=0)
    td = TestDiagnostic()
    simulation.diagnostics[:td] = td
    return simulation.diagnostics[1] == td
end

function diagnostics_setindex(arch, FT)
    model = TestModel_RegularRectGrid(arch, FT)
    simulation = Simulation(model, Δt=0, stop_iteration=0)

    td1 = TestDiagnostic()
    td2 = TestDiagnostic()
    td3 = TestDiagnostic()

    push!(simulation.diagnostics, td1, td2)
    simulation.diagnostics[2] = td3

    return simulation.diagnostics[:diag2] == td3
end

@testset "Diagnostics" begin
    @info "Testing diagnostics..."

    for arch in archs
        @testset "CFL [$(typeof(arch))]" begin
            @info "  Testing CFL diagnostics [$(typeof(arch))]..."
            for FT in float_types
                @test diffusive_cfl_diagnostic_is_correct(arch, FT)
                @test advective_cfl_diagnostic_is_correct_on_regular_grid(arch, FT)
                @test advective_cfl_diagnostic_is_correct_on_vertically_stretched_grid(arch, FT)
                @test accurate_advective_cfl_on_regular_grid(arch, FT)
                @test accurate_advective_cfl_on_stretched_grid(arch, FT)
                @test accurate_advective_cfl_on_lat_lon_grid(arch, FT)
            end
        end
    end

    for arch in archs
        @testset "Miscellaneous timeseries diagnostics [$(typeof(arch))]" begin
            @info "  Testing miscellaneous timeseries diagnostics [$(typeof(arch))]..."
            for FT in float_types
                @test diagnostics_getindex(arch, FT)
                @test diagnostics_setindex(arch, FT)
            end
        end
    end
end
