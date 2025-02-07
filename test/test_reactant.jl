using Reactant

if haskey(ENV, "GPU_TEST")
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

using Test
using Oceananigans
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using GPUArrays
using Random

function test_reactant_model_correctness(GridType, ModelType, grid_kw, model_kw)
    r_arch = ReactantState()
    r_grid = GridType(r_arch; grid_kw...)
    r_model = ModelType(; grid=r_grid, model_kw...)

    grid = GridType(CPU(); grid_kw...)
    model = ModelType(; grid=grid, model_kw...)

    ui(x, y, z) = randn()

    Random.seed!(123)
    set!(model, u=ui, v=ui)

    Random.seed!(123)
    set!(r_model, u=ui, v=ui)

    # Deduce a stable time-step
    Δx = minimum_xspacing(grid)
    Δt = 0.1 / Δx

    # Stop iteration for both simulations
    stop_iteration = 3

    # What we normally do:
    simulation = Simulation(model; Δt, stop_iteration)
    run!(simulation)

    # What we want to do with Reactant:
    r_simulation = Simulation(r_model; Δt, stop_iteration)
    pop!(r_simulation.callbacks, :nan_checker)

    r_run! = @compile sync = true run!(r_simulation)
    r_run!(r_simulation)

    # Some tests
    # Things ran normally:
    @test iteration(r_simulation) == iteration(simulation)
    @test time(r_simulation) == time(simulation)

    # Data looks right:
    u, v, w = model.velocities
    ru, rv, rw = r_model.velocities

    @show maximum(abs.(parent(u) .- parent(ru)))
    @show maximum(abs.(parent(v) .- parent(rv)))
    @show maximum(abs.(parent(w) .- parent(rw)))

    @test parent(u) ≈ parent(ru)
    @test parent(v) ≈ parent(rv)
    @test parent(w) ≈ parent(rw)

    return nothing
end

@testset "Reactanigans unit tests" begin
    arch = ReactantState()
    grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1))
    c = CenterField(grid)
    @test parent(c) isa Reactant.ConcreteRArray

    set!(c, (x, y, z) -> x + y * z)
    x, y, z = nodes(c)

    @allowscalar begin
        @test c[1, 1, 1] == x[1] + y[1] * z[1]
        @test c[1, 2, 1] == x[1] + y[2] * z[1]
        @test c[1, 2, 3] == x[1] + y[2] * z[3]
    end

    @jit fill_halo_regions!(c)

    @allowscalar begin
        @test c[1, 1, 0] == c[1, 1, 1]
    end

    d = CenterField(grid)
    parent(d) .= 2

    cd = Field(c * d)
    compute!(cd)

    @allowscalar begin
        @test cd[1, 1, 1] == 2 * (x[1] + y[1] * z[1])
        @test cd[1, 2, 1] == 2 * (x[1] + y[2] * z[1])
        @test cd[1, 2, 3] == 2 * (x[1] + y[2] * z[3])
    end
end

@testset "Reactant Super Simple Simulation Tests" begin
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    longitude = (0, 4)
    latitude = (0, 4)
    z = (-1, 0)

    nonhydrostatic_model_kw = (; advection=WENO())
    hydrostatic_model_kw = (; momentum_advection=WENO())
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude, latitude, z)

    # FFTs are not supported by Reactant so we don't run this test:
    # @info "Testing RectilinearGrid + NonhydrostaticModel Reactant correctness"
    # test_reactant_model_correctness(RectilinearGrid, NonhydrostaticModel, rectilinear_kw, nonhydrostatic_model_kw)

    @info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    hydrostatic_model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))
    test_reactant_model_correctness(RectilinearGrid, HydrostaticFreeSurfaceModel, rectilinear_kw, hydrostatic_model_kw)

    @info "Testing LatitudeLongitudeGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    hydrostatic_model_kw = (; momentum_advection=WENO())
    test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw)

    hydrostatic_model_kw = (; momentum_advection=WENOVectorInvariant(), closure=CATKEVerticalDiffusivity())
    test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw)
end

