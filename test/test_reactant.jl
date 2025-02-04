using Reactant
using Test
using Oceananigans
using Oceananigans.Architectures
using GPUArrays
GPUArrays.allowscalar(true)

@testset "Reactant Super Simple Simulation Tests" begin
    r_arch = ReactantState()
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    longitude = (0, 4)
    latitude = (0, 4)
    z = (-1000, 0)

    r_grid = LatitudeLongitudeGrid(r_arch; size=(Nx, Ny, Nz), halo, longitude, latitude, z)
    grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz), halo, longitude, latitude, z)

    r_model = HydrostaticFreeSurfaceModel(; grid=r_grid, momentum_advection=WENO())
    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENO())

    uᵢ(x, y, z) = randn()
    Random.seed!(123)
    set!(r_model, u=uᵢ, v=uᵢ)
    Random.seed!(123)
    set!(model, u=uᵢ, v=uᵢ)

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
end
