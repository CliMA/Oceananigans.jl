using MPI
using Random
using Test
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!, UPivot, TPivot, FPivot

rectilinear_grid(arch, topology) = RectilinearGrid(arch; size = (16, 16, 4), extent = (1, 1, 1), halo = (3, 3, 3), topology)

function latitude_longitude_grid(arch, topology)
    longitude = topology[1] == Periodic ? (0, 360) : (0, 90)
    return LatitudeLongitudeGrid(arch; size = (16, 16, 4), halo = (3, 3, 3), longitude, latitude = (0, 10), z = (0, 1), topology)
end

tripolar_grid(arch, (fold_topology, pivot)) = TripolarGrid(arch; size = (16, 20, 4), halo = (3, 3, 3), z = (0, 1), fold_topology, pivot)

mpi_reactant_architecture(partition) = Distributed(ReactantState(); partition, communicator = MPI.COMM_WORLD)

"""
    test_mpi_fill_halo_regions(make_grid, partition, topology; traced=true)

Test that one `fill_halo_regions!` of fields at every location matches `Distributed(child_arch)` on each rank.
The fill is compiled if `traced` and called directly on the concrete arrays otherwise.
"""
function test_mpi_fill_halo_regions(make_grid, partition, topology; traced=true)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    vanilla_grid = make_grid(Distributed(child_arch; partition), topology)
    reactant_grid = make_grid(mpi_reactant_architecture(partition), topology)
    locations = vec(collect(Iterators.product((Center, Face), (Center, Face), (Center, Face))))

    vanilla_fields = Tuple(Field{LX, LY, LZ}(vanilla_grid) for (LX, LY, LZ) in locations)
    reactant_fields = Tuple(Field{LX, LY, LZ}(reactant_grid) for (LX, LY, LZ) in locations)

    Random.seed!(12345 + rank)
    for (vanilla_field, reactant_field) in zip(vanilla_fields, reactant_fields)
        data = randn(size(vanilla_field)...)
        set!(vanilla_field, data)
        set!(reactant_field, data)
    end

    fill_halo_regions!(vanilla_fields)

    if traced
        rfill! = @compile raise=true raise_first=true fill_halo_regions!(reactant_fields)
        rfill!(reactant_fields)
    else
        fill_halo_regions!(reactant_fields)
    end

    for (location, vanilla_field, reactant_field) in zip(locations, vanilla_fields, reactant_fields)
        @test compare_parent("loc=$location", vanilla_field, reactant_field)
    end

    return nothing
end

"""
    test_mpi_compute_simple_Gu(make_grid, topology, coriolis)

Test that `compute_simple_Gu!` with WENO, which reads every halo cell in both horizontal directions,
matches `Distributed(child_arch)` on each rank of a `Partition(2, 2)`.
"""
function test_mpi_compute_simple_Gu(make_grid, topology, coriolis)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    partition = Partition(2, 2)
    vanilla_grid = make_grid(Distributed(child_arch; partition), topology)
    reactant_grid = make_grid(mpi_reactant_architecture(partition), topology)

    vanilla_velocities = (u = XFaceField(vanilla_grid), v = YFaceField(vanilla_grid), w = ZFaceField(vanilla_grid))
    reactant_velocities = (u = XFaceField(reactant_grid), v = YFaceField(reactant_grid), w = ZFaceField(reactant_grid))

    Random.seed!(54321 + rank)
    for (vanilla_velocity, reactant_velocity) in zip(vanilla_velocities, reactant_velocities)
        data = randn(size(vanilla_velocity)...)
        set!(vanilla_velocity, data)
        set!(reactant_velocity, data)
    end

    fill_halo_regions!(vanilla_velocities)
    rfill! = @compile raise=true raise_first=true fill_halo_regions!(reactant_velocities)
    rfill!(reactant_velocities)

    advection = Oceananigans.Advection.materialize_advection(WENO(), reactant_grid)
    vanilla_Gu = XFaceField(vanilla_grid)
    reactant_Gu = XFaceField(reactant_grid)
    compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)

    # raise=true is broken with WENO and a Bounded x-axis
    rcompute_Gu! = if topology[1] == Bounded
        @compile compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
    else
        @compile raise=true raise_first=true compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
    end
    rcompute_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
    @test compare_interior("Gu", vanilla_Gu, reactant_Gu)

    return nothing
end

"""
    test_mpi_hydrostatic_free_surface_model(partition, topology; Nx=16, Ny=16, Nz=4, stop_iteration=3)

Test that compiled `first_time_step!` and `time_step!` of a `HydrostaticFreeSurfaceModel` with a
`SplitExplicitFreeSurface`, WENO advection, and a TEOS10 equation of state match `Distributed(child_arch)`
on each rank after `stop_iteration` steps.
"""
function test_mpi_hydrostatic_free_surface_model(partition, topology; Nx=16, Ny=16, Nz=4, stop_iteration=3)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    grid_kw = (size = (Nx, Ny, Nz), halo = (6, 6, 3), x = (0, 1e5), y = (0, 1e5), z = (-100, 0), topology)
    vanilla_grid = RectilinearGrid(Distributed(child_arch; partition); grid_kw...)
    reactant_grid = RectilinearGrid(mpi_reactant_architecture(partition); grid_kw...)

    model_kw = (coriolis = FPlane(f=1e-4),
                buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState()),
                free_surface = SplitExplicitFreeSurface(vanilla_grid; substeps = 10),
                tracers = (:T, :S),
                tracer_advection = WENO(),
                momentum_advection = WENOVectorInvariant(),
                closure = nothing)

    vanilla_model = HydrostaticFreeSurfaceModel(vanilla_grid; model_kw...)
    reactant_model = HydrostaticFreeSurfaceModel(reactant_grid; model_kw...)

    # Initial conditions are sized to the local grid
    Random.seed!(98765 + rank)
    lx, ly, lz = size(vanilla_grid)
    u_init = 0.1 * randn(size(vanilla_model.velocities.u)...)
    v_init = 0.1 * randn(size(vanilla_model.velocities.v)...)
    T_init = [20.0 + 5.0 * (k - 0.5) / lz + 0.01 * randn() for i = 1:lx, j = 1:ly, k = 1:lz]
    S_init = [35.0 + 0.01 * randn() for i = 1:lx, j = 1:ly, k = 1:lz]

    set!(vanilla_model, u=u_init, v=v_init, T=T_init, S=S_init)
    set!(reactant_model, u=u_init, v=v_init, T=T_init, S=S_init)

    Δt = 60.0
    vanilla_simulation = Simulation(vanilla_model; Δt, stop_iteration, verbose=false)
    run!(vanilla_simulation)

    @jit Oceananigans.TimeSteppers.update_state!(reactant_model)
    r_first_time_step! = @compile sync=true Oceananigans.TimeSteppers.first_time_step!(reactant_model, Δt)
    r_time_step! = @compile sync=true Oceananigans.TimeSteppers.time_step!(reactant_model, Δt)

    reactant_simulation = Simulation(reactant_model; Δt, stop_iteration, verbose=false)
    r_run!(reactant_simulation, r_time_step!, r_first_time_step!)

    @test iteration(reactant_simulation) == iteration(vanilla_simulation)
    @test time(reactant_simulation) ≈ time(vanilla_simulation)
    @test compare_interior("u", vanilla_model.velocities.u, reactant_model.velocities.u)
    @test compare_interior("v", vanilla_model.velocities.v, reactant_model.velocities.v)
    @test compare_interior("T", vanilla_model.tracers.T, reactant_model.tracers.T)
    @test compare_interior("S", vanilla_model.tracers.S, reactant_model.tracers.S)

    return nothing
end
