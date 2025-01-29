using Reactant

using Oceananigans
using Reactant

@testset "Reactant Super Simple Simulation Tests" begin
    r_arch = ReactantState()
    Nx, Ny, Nz = (360, 120, 100) # number of cells

    r_grid = LatitudeLongitudeGrid(r_arch, size=(Nx, Ny, Nz), halo=(7, 7, 7),
                                 longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0))
    
    arch = CPU()
    grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7),
                                 longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0))

    # One of the implest configurations we might consider:
    r_model = HydrostaticFreeSurfaceModel(; r_grid, momentum_advection=WENO())
    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENO())

    @assert r_model.free_surface isa SplitExplicitFreeSurface
    @assert model.free_surface isa SplitExplicitFreeSurface

    uᵢ(x, y, z) = randn()
    set!(r_model, u=uᵢ, v=uᵢ)
    set!(model, u=uᵢ, v=uᵢ)

    # Deduce a stable time-step
    Δx = minimum_xspacing(grid)
    Δt = 0.1 / Δx

    # Stop iteration for both simulations
    stop_iteration = 100

    # First form a Reactant model
    @assert typeof(Reactant.to_rarray(model)) == typeof(r_model)

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

    @test parent(u) == parent(ru)
    @test parent(v) == parent(rv)
    @test parent(w) == parent(rw)
end