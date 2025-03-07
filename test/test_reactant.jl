using Reactant

if haskey(ENV, "GPU_TEST")
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

using Test
using OffsetArrays
using Oceananigans
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.Utils: launch!
using SeawaterPolynomials: TEOS10EquationOfState
using KernelAbstractions: @kernel, @index
using Random

OceananigansReactantExt = Base.get_extension(Oceananigans, :OceananigansReactantExt)

bottom_height(x, y) = - 0.5

function test_reactant_model_correctness(GridType, ModelType, grid_kw, model_kw; immersed_boundary_grid=true)
    r_arch = ReactantState()
    r_grid = GridType(r_arch; grid_kw...)
    grid = GridType(CPU(); grid_kw...)

    if immersed_boundary_grid
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
        r_grid = ImmersedBoundaryGrid(r_grid, GridFittedBottom(bottom_height))
        @test isnothing(r_grid.interior_active_cells)
        @test isnothing(r_grid.active_z_columns)
        @test isnothing(grid.interior_active_cells)
        @test isnothing(grid.active_z_columns)
    end

    r_model = ModelType(; grid=r_grid, model_kw...)
    model = ModelType(; grid=grid, model_kw...)

    ui = randn(size(model.velocities.u)...)
    vi = randn(size(model.velocities.v)...)

    set!(model, u=ui, v=ui)
    set!(r_model, u=ui, v=ui)

    u, v, w = model.velocities
    ru, rv, rw = r_model.velocities

    # Test that fields were set correctly
    @info "  After setting an initial condition:"
    @show maximum(abs.(parent(u) .- parent(ru)))
    @show maximum(abs.(parent(v) .- parent(rv)))
    @show maximum(abs.(parent(w) .- parent(rw)))

    @test parent(u) ≈ parent(ru)
    @test parent(v) ≈ parent(rv)
    @test parent(w) ≈ parent(rw)

    # Deduce a stable time-step
    Δx = minimum_xspacing(grid)
    Δt = 0.01 * Δx

    # Stop iteration for both simulations
    stop_iteration = 3

    # What we normally do:
    simulation = Simulation(model; Δt, stop_iteration, verbose=false)
    run!(simulation)

    # What we want to do with Reactant:
    r_simulation = Simulation(r_model; Δt, stop_iteration, verbose=false)
    pop!(r_simulation.callbacks, :nan_checker)

    r_run! = @compile sync = true run!(r_simulation)
    r_run!(r_simulation)

    # Some tests
    # Things ran normally:
    @test iteration(r_simulation) == iteration(simulation)
    @test time(r_simulation) == time(simulation)

    @info "  After running 3 time steps:"
    @show maximum(abs, parent(u))
    @show maximum(abs, parent(v))
    @show maximum(abs, parent(w))

    @show maximum(abs.(parent(u) .- parent(ru)))
    @show maximum(abs.(parent(v) .- parent(rv)))
    @show maximum(abs.(parent(w) .- parent(rw)))

    @test parent(u) ≈ parent(ru)
    @test parent(v) ≈ parent(rv)
    @test parent(w) ≈ parent(rw)

    return r_simulation
end

function add_one!(f)
    arch = architecture(f)
    launch!(arch, f.grid, :xyz, _add_one!, f)
    return f
end

@kernel function _add_one!(f)
    i, j, k = @index(Global, NTuple)
    @inbounds f[i, j, k] += 1
end

@testset "Reactanigans unit tests" begin
    @info "Performing Reactanigans unit tests..."
    arch = ReactantState()
    grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1))
    c = CenterField(grid)
    @test parent(c) isa Reactant.ConcretePJRTArray


    @info "  Testing field set! with a number..."
    set!(c, 1)
    @test all(c .≈ 1)

    @info "  Testing simple kernel launch!..."
    add_one!(c)
    @test all(c .≈ 2)

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

    # Deconcretization
    c′ = OceananigansReactantExt.deconcretize(c)
    @test parent(c′) isa Array
    @test architecture(c′) isa ReactantState

    for FT in (Float64, Float32)
        sgrid = RectilinearGrid(arch, FT; size=(4, 4, 4), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 1))
        @test architecture(sgrid) isa ReactantState
        @test architecture(sgrid.xᶠᵃᵃ) isa CPU
        @test architecture(sgrid.xᶜᵃᵃ) isa CPU

        llg = LatitudeLongitudeGrid(arch, FT; size = (4, 4, 4),
                                    longitude = [0, 1, 2, 3, 4],
                                    latitude = [0, 1, 2, 3, 4],
                                    z = (0, 1))

        @test architecture(llg) isa ReactantState

        for name in propertynames(llg)
            p = getproperty(llg, name)
            if !(name ∈ (:architecture, :z))
                @test (p isa Number) || (p isa OffsetArray{FT, <:Any, <:Array})
            end
        end

        ridge(λ, φ) = 0.1 * exp((λ - 2)^2 / 2)
        ibg = ImmersedBoundaryGrid(llg, GridFittedBottom(ridge))
        @test architecture(ibg) isa ReactantState
        @test architecture(ibg.immersed_boundary.bottom_height) isa CPU

        rllg = RotatedLatitudeLongitudeGrid(arch, FT; size = (4, 4, 4),
                                            north_pole = (0, 0),
                                            longitude = [0, 1, 2, 3, 4],
                                            latitude = [0, 1, 2, 3, 4],
                                            z = (0, 1))

        @test architecture(rllg) isa ReactantState

        for name in propertynames(rllg)
            p = getproperty(rllg, name)
            if !(name ∈ (:architecture, :z, :conformal_mapping))
                @test (p isa Number) || (p isa OffsetArray{FT, <:Any, <:Array})
            end
        end
    end
end

@testset "Reactant Super Simple Simulation Tests" begin
    @info "Performing Reactanigans super simple simulation tests..."
    nonhydrostatic_model_kw = (; advection=WENO())
    hydrostatic_model_kw = (; momentum_advection=WENO())
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    longitude = (0, 4)
    stretched_longitude = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.3, 2.5, 2.6, 3.5, 4.0]
    latitude = (0, 4)
    z = (-1, 0)
    lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude, latitude, z)
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    stretched_lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude=stretched_longitude, latitude, z)

    # FFTs are not supported by Reactant so we don't run this test:
    # @info "Testing RectilinearGrid + NonhydrostaticModel Reactant correctness"
    # test_reactant_model_correctness(RectilinearGrid, NonhydrostaticModel, rectilinear_kw, nonhydrostatic_model_kw)

    @info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    hydrostatic_model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))
    test_reactant_model_correctness(RectilinearGrid, HydrostaticFreeSurfaceModel, rectilinear_kw, hydrostatic_model_kw)
    test_reactant_model_correctness(RectilinearGrid, HydrostaticFreeSurfaceModel, rectilinear_kw, hydrostatic_model_kw, immersed_boundary_grid=true)

    @info "Testing LatitudeLongitudeGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    hydrostatic_model_kw = (; momentum_advection=WENO())
    test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw)
    test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw, immersed_boundary_grid=true)

    #=
    # This test takes too long
    @info "Testing LatitudeLongitudeGrid + SplitExplicitFreeSurface + HydrostaticFreeSurfaceModel Reactant correctness"
    hydrostatic_model_kw = (; momentum_advection=WENOVectorInvariant(), free_surface=SplitExplicitFreeSurface(substeps=4))
    test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw)
    simulation = test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw, immersed_boundary_grid=true)
    η = simulation.model.free_surface.η
    η_grid = η.grid
    @test isnothing(η_grid.interior_active_cells)
    @test isnothing(η_grid.active_z_columns)
    =#

    #=
    equation_of_state = TEOS10EquationOfState()
    hydrostatic_model_kw = (momentum_advection = WENOVectorInvariant(),
                            tracer_advection = WENO(),
                            tracers = (:T, :S, :e),
                            buoyancy = SeawaterBuoyancy(; equation_of_state),
                            closure = CATKEVerticalDiffusivity())
    test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw)
    =#
end

@testset "Reactanigans Clock{ConcreteRNumber} tests" begin
    @info "Testing model time-stepping with Clock{ConcreteRNumber}..."

    # All of these may not need to be traced but this is paranoia.
    FT = Float64
    t = ConcreteRNumber(zero(FT))
    iter = ConcreteRNumber(0)
    stage = ConcreteRNumber(0)
    last_Δt = ConcreteRNumber(zero(FT))
    last_stage_Δt = ConcreteRNumber(zero(FT))
    clock = Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)

    grid = RectilinearGrid(ReactantState(); size=(10, 10, 10), halo=(3, 3, 3), extent=(10, 10, 10))
    free_surface = SplitExplicitFreeSurface(grid, substeps=10, gravitational_acceleration=1)
    model = HydrostaticFreeSurfaceModel(; grid, clock, free_surface)

    Δt = 0.02
    simulation = Simulation(model; Δt, stop_iteration=3, verbose=false)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 0.06

    simulation.stop_iteration += 2
    run!(simulation)
    @test iteration(simulation) == 5
    @test time(simulation) == 0.10
end

