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
using GPUArrays

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

    fill_halo_regions!(c)

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

    @test_broken parent(u) ≈ parent(ru)
    @test_broken parent(v) ≈ parent(rv)
    @test_broken parent(w) ≈ parent(rw)
end

