include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))

using CUDA

@kernel function _simple_tendency_kernel!(Gu, grid, advection, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = - Oceananigans.Advection.U_dot_∇u(i, j, k, grid, advection, velocities)
end

function simple_tendency!(model)
    grid = model.grid
    arch = grid.architecture
    Oceananigans.Utils.launch!(
        arch,
        grid,
        :xyz,
        _simple_tendency_kernel!,
        model.timestepper.Gⁿ.u,
        grid,
        model.advection.momentum,
        model.velocities)
    return nothing
end

@testset "Gu kernel" begin
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    longitude = (0, 4)
    latitude = (0, 4)
    z = (-1, 0)
    lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude, latitude, z)
    hydrostatic_model_kw = (; momentum_advection=VectorInvariant(), free_surface=ExplicitFreeSurface())

    arch = Oceananigans.Architectures.ReactantState()
    grid = LatitudeLongitudeGrid(arch; lat_lon_kw...)
    model = HydrostaticFreeSurfaceModel(grid; hydrostatic_model_kw...)

    @test model.clock.stage == 1

    ui = randn(size(model.velocities.u)...)
    vi = randn(size(model.velocities.v)...)
    set!(model, u=ui, v=vi)

    @jit simple_tendency!(model)

    Gu = model.timestepper.Gⁿ.u
    Gv = model.timestepper.Gⁿ.v
    Gui = Array(interior(Gu))
    Gvi = Array(interior(Gv))

    carch = CPU()
    cgrid = LatitudeLongitudeGrid(carch; lat_lon_kw...)
    cmodel = HydrostaticFreeSurfaceModel(cgrid; hydrostatic_model_kw...)

    set!(cmodel, u=ui, v=vi)

    simple_tendency!(cmodel)
    @test all(Gui .≈ Array(interior(cmodel.timestepper.Gⁿ.u)))
    @test all(Gvi .≈ Array(interior(cmodel.timestepper.Gⁿ.v)))
end

@testset "Reactant RectilinearGrid Simulation Tests" begin
    @info "Performing Reactanigans RectilinearGrid simulation tests..."
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    z = (-1, 0)
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    hydrostatic_model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))
    rungekutta3_kw = merge(hydrostatic_model_kw, (; timestepper=:SplitRungeKutta3))

    @info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    hydrostatic_model_kw)

    @info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel + SplitRungeKutta3 Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    rungekutta3_kw)

    @info "Testing immersed RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    hydrostatic_model_kw,
                                    immersed_boundary_grid=true)

    @info "Testing immersed RectilinearGrid + HydrostaticFreeSurfaceModel + SplitRungeKutta3 Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    rungekutta3_kw,
                                    immersed_boundary_grid=true)
end

@testset "Reactant Simulation: compiled run! with callbacks" begin
    @info "Testing a compiled run! of a Reactant Simulation with an IterationInterval callback..."
    Nx, Ny, Nz = (10, 10, 10)
    halo = (7, 7, 7)
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))

    grid = RectilinearGrid(CPU(); rectilinear_kw...)
    r_grid = RectilinearGrid(ReactantState(); rectilinear_kw...)
    model = HydrostaticFreeSurfaceModel(grid; model_kw...)
    r_model = HydrostaticFreeSurfaceModel(r_grid; model_kw...)

    ui = randn(size(model.velocities.u)...)
    vi = randn(size(model.velocities.v)...)
    set!(model, u=ui, v=vi)
    set!(r_model, u=ui, v=vi)

    Δt = 1e-6 * minimum_xspacing(grid)
    stop_iteration = 4

    # A callback that does only device work: accumulate Σu² into a length-one array every second
    # step. The eager run fires it at iteration 0 (initialization) and at iterations 2 and 4; the
    # compiled run must do the same.
    accumulate_u²!(sim, p) = (p.total .+= sum(interior(sim.model.velocities.u) .^ 2); nothing)
    energy_callback(total) = Callback(accumulate_u²!, IterationInterval(2); parameters=(; total))

    simulation = Simulation(model; Δt, stop_iteration, verbose=false)
    total = zeros(1)
    add_callback!(simulation, energy_callback(total); name=:energy)
    run!(simulation)

    r_simulation = Simulation(r_model; Δt, stop_iteration, verbose=false)
    r_total = Reactant.to_rarray(zeros(1))
    add_callback!(r_simulation, energy_callback(r_total); name=:energy)
    compiled_run! = @compile run!(r_simulation)
    compiled_run!(r_simulation)

    @test iteration(r_simulation) == stop_iteration
    @test Array(interior(r_model.velocities.u)) ≈ Array(interior(model.velocities.u))
    @test Array(interior(r_model.velocities.v)) ≈ Array(interior(model.velocities.v))
    @test Array(r_total)[1] ≈ total[1]
    @test total[1] > 0
end

@testset "Reactant Simulation: stop_time and TimeInterval" begin
    @info "Testing stop_time and TimeInterval conversion on a Reactant Simulation..."
    Nx, Ny, Nz = (10, 10, 10)
    halo = (7, 7, 7)
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))

    function fresh_models()
        grid = RectilinearGrid(CPU(); rectilinear_kw...)
        r_grid = RectilinearGrid(ReactantState(); rectilinear_kw...)
        model = HydrostaticFreeSurfaceModel(grid; model_kw...)
        r_model = HydrostaticFreeSurfaceModel(r_grid; model_kw...)
        ui = randn(size(model.velocities.u)...)
        vi = randn(size(model.velocities.v)...)
        set!(model, u=ui, v=vi)
        set!(r_model, u=ui, v=vi)
        return model, r_model
    end

    accumulate_u²!(sim, p) = (p.total .+= sum(interior(sim.model.velocities.u) .^ 2); nothing)

    model, r_model = fresh_models()
    Δt = 1e-6 * minimum_xspacing(model.grid)

    # stop_time that Δt divides, and a TimeInterval callback that Δt divides: the eager and the
    # compiled runs take the same 4 steps and fire at iterations 0, 2, 4.
    simulation = Simulation(model; Δt, stop_time=4Δt, verbose=false)
    total = zeros(1)
    add_callback!(simulation, accumulate_u²!, TimeInterval(2Δt); parameters=(; total), name=:energy)
    run!(simulation)

    r_simulation = Simulation(r_model; Δt, stop_time=4Δt, verbose=false)
    @test r_simulation.stop_iteration == 4
    @test isnothing(r_simulation.stop_time)
    r_total = Reactant.to_rarray(zeros(1))
    add_callback!(r_simulation, accumulate_u²!, TimeInterval(2Δt); parameters=(; total=r_total), name=:energy)
    @test r_simulation.callbacks[:energy].schedule isa IterationInterval
    @test r_simulation.callbacks[:energy].schedule.interval == 2

    compiled_run! = @compile run!(r_simulation)
    compiled_run!(r_simulation)

    @test iteration(r_simulation) == 4
    @test Reactant.to_number(r_model.clock.time) ≈ 4Δt
    @test Array(interior(r_model.velocities.u)) ≈ Array(interior(model.velocities.u))
    @test Array(r_total)[1] ≈ total[1]

    # stop_time that Δt does not divide: a warning at construction, 3 whole steps, and one
    # remainder step of Δt/2, landing where the eager aligned last step lands.
    model, r_model = fresh_models()
    simulation = Simulation(model; Δt, stop_time=3.5Δt, verbose=false)
    run!(simulation)

    r_simulation = @test_logs (:warn, r"does not divide") Simulation(r_model; Δt, stop_time=3.5Δt, verbose=false)
    @test r_simulation.stop_iteration == 3
    @test r_simulation.stop_time isa OceananigansReactantExt.Simulations.RemainderStep
    @test r_simulation.stop_time.Δt ≈ 0.5Δt

    compiled_run! = @compile run!(r_simulation)
    compiled_run!(r_simulation)

    @test iteration(r_simulation) == 4
    @test Reactant.to_number(r_model.clock.time) ≈ 3.5Δt
    @test Array(interior(r_model.velocities.u)) ≈ Array(interior(model.velocities.u))

    # Schedules a program cannot evaluate are refused when added.
    @test_throws ArgumentError add_callback!(r_simulation, accumulate_u²!, TimeInterval(2.5Δt); parameters=(; total=r_total))
    @test_throws ArgumentError add_callback!(r_simulation, accumulate_u²!, WallTimeInterval(1.0); parameters=(; total=r_total))
    @test_throws ArgumentError Simulation(r_model; Δt, stop_iteration=4, stop_time=4Δt, verbose=false)
end

# A callback whose function type records the initial and final Σu² through the `initialize!` and
# `finalize!` hooks, which run once each outside the traced loop. It carries its own buffer, since
# the hooks receive the function, not the callback's parameters.
struct EnergyBookends{T}
    values :: T   # [initial, final]
end
(::EnergyBookends)(sim) = nothing
Σu²(sim) = sum(interior(sim.model.velocities.u) .^ 2)
Oceananigans.initialize!(bookends::EnergyBookends, sim) = (bookends.values[1:1] .= Σu²(sim); nothing)
Oceananigans.Simulations.finalize!(bookends::EnergyBookends, sim) = (bookends.values[2:2] .= Σu²(sim); nothing)

@testset "Reactant Simulation: initialize! and finalize! on callbacks" begin
    @info "Testing the initialize!/finalize! callback hooks under a compiled run!..."
    Nx, Ny, Nz = (10, 10, 10)
    halo = (7, 7, 7)
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))

    grid = RectilinearGrid(CPU(); rectilinear_kw...)
    r_grid = RectilinearGrid(ReactantState(); rectilinear_kw...)
    model = HydrostaticFreeSurfaceModel(grid; model_kw...)
    r_model = HydrostaticFreeSurfaceModel(r_grid; model_kw...)
    ui = randn(size(model.velocities.u)...)
    vi = randn(size(model.velocities.v)...)
    set!(model, u=ui, v=vi)
    set!(r_model, u=ui, v=vi)

    Δt = 1e-6 * minimum_xspacing(grid)
    stop_iteration = 4

    simulation = Simulation(model; Δt, stop_iteration, verbose=false)
    bookends = EnergyBookends(zeros(2))
    add_callback!(simulation, Callback(bookends, IterationInterval(1)); name=:bookends)
    run!(simulation)

    r_simulation = Simulation(r_model; Δt, stop_iteration, verbose=false)
    r_bookends = EnergyBookends(Reactant.to_rarray(zeros(2)))
    add_callback!(r_simulation, Callback(r_bookends, IterationInterval(1)); name=:bookends)
    compiled_run! = @compile run!(r_simulation)
    compiled_run!(r_simulation)

    r_values = Array(r_bookends.values)
    @test r_values[1] ≈ bookends.values[1]           # initialize!: the initial state
    @test r_values[2] ≈ bookends.values[2]           # finalize!: the final state
    @test r_values[2] ≈ sum(Array(interior(r_model.velocities.u)) .^ 2)
    @test r_values[1] != r_values[2]
end
