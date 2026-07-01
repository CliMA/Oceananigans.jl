include("dependencies_for_runtests.jl")

using JLD2
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, ExplicitFreeSurface
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

#####
##### Distributed output combining tests
#####
#
# These tests verify that distributed output (with _rank0, _rank1, etc. suffixes)
# can be automatically combined into a global FieldTimeSeries that matches
# output from an equivalent non-distributed simulation.
#

#####
##### Helper functions
#####

function run_mpi_script(script, filename, nranks=4)
    write(filename, script)
    run(`$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 $filename`)
    rm(filename)
end

function cleanup_rank_files(prefix, nranks=4)
    for r in 0:(nranks-1)
        rm("$(prefix)_rank$r.jld2", force=true)
    end
end

# Custom comparison that handles NaN values (NaN == NaN for our purposes)
function arrays_equal_with_nans(a, b; rtol=√eps())
    size(a) != size(b) && return false
    for (x, y) in zip(a, b)
        if isnan(x) && isnan(y)
            continue
        elseif isnan(x) || isnan(y)
            return false
        elseif !isapprox(x, y; rtol)
            return false
        end
    end
    return true
end

function test_combined_output_matches_serial(distributed_prefix, serial_file, varnames)
    for varname in varnames
        c_distributed = FieldTimeSeries("$distributed_prefix.jld2", varname)
        c_serial = FieldTimeSeries(serial_file, varname)

        @test size(c_distributed.grid) == size(c_serial.grid)
        @test size(c_distributed) == size(c_serial)
        @test c_distributed.times ≈ c_serial.times

        for n in 1:length(c_distributed.times)
            @test arrays_equal_with_nans(interior(c_distributed[n]), interior(c_serial[n]))
        end
    end
end

#####
##### RectilinearGrid test configuration
#####

const rectilinear_2x2_config = (
    size = (8, 8, 4),
    extent = (1.0, 1.0, 0.5),
    partition = (2, 2),
    Δt = 1.0,
    stop_iteration = 10,
    output_interval = 5,
)

const rectilinear_slab_config = (
    size = (8, 16, 4),
    extent = (1.0, 2.0, 0.5),
    partition = (1, 4),
    Δt = 1.0,
    stop_iteration = 4,
    output_interval = 2,
)

function rectilinear_mpi_script(config, filename)
    Nx, Ny, Nz = config.size
    Lx, Ly, Lz = config.extent
    px, py = config.partition
    Δt = config.Δt
    stop_iteration = config.stop_iteration
    output_interval = config.output_interval

    return """
    using MPI
    MPI.Init()

    using Oceananigans
    using Oceananigans.DistributedComputations: Distributed, Partition

    arch = Distributed(CPU(), partition=Partition($px, $py))

    grid = RectilinearGrid(arch;
                           topology = (Periodic, Periodic, Bounded),
                           size = ($Nx, $Ny, $Nz),
                           extent = ($Lx, $Ly, $Lz))

    model = NonhydrostaticModel(grid; tracers=:c)

    Lx, Ly, Lz = $Lx, $Ly, $Lz
    cᵢ(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly) * (z + Lz) / Lz
    uᵢ(x, y, z) = 0.1 * sin(2π * x / Lx)
    set!(model, c=cᵢ, u=uᵢ)

    # Add a `Nothing field in the z-direction`
    zflat = Field{Center, Center, Nothing}(grid)
    set!(zflat, (x, y) -> x)

    simulation = Simulation(model; Δt=$Δt, stop_iteration=$stop_iteration)

    simulation.output_writers[:jld2] = JLD2Writer(model,
                                                  merge(model.velocities, model.tracers, (; zflat));
                                                  filename = "$filename",
                                                  schedule = IterationInterval($output_interval),
                                                  overwrite_existing = true,
                                                  with_halos = true)
    run!(simulation)

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
    """
end

function run_serial_rectilinear(config, filename)
    Nx, Ny, Nz = config.size
    Lx, Ly, Lz = config.extent

    grid = RectilinearGrid(CPU();
                           topology = (Periodic, Periodic, Bounded),
                           size = (Nx, Ny, Nz),
                           extent = (Lx, Ly, Lz))

    model = NonhydrostaticModel(grid; tracers=:c)

    cᵢ(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly) * (z + Lz) / Lz
    uᵢ(x, y, z) = 0.1 * sin(2π * x / Lx)
    set!(model, c=cᵢ, u=uᵢ)

    # Add a `Nothing field in the z-direction`
    zflat = Field{Center, Center, Nothing}(grid)
    set!(zflat, (x, y) -> x)

    simulation = Simulation(model; Δt=config.Δt, stop_iteration=config.stop_iteration)

    simulation.output_writers[:jld2] = JLD2Writer(model,
                                                  merge(model.velocities, model.tracers, (; zflat));
                                                  filename = filename,
                                                  schedule = IterationInterval(config.output_interval),
                                                  overwrite_existing = true,
                                                  with_halos = true)
    run!(simulation)
end

#####
##### LatitudeLongitudeGrid test configuration
#####

const lat_lon_config = (
    size = (16, 16, 4),
    longitude = (-30, 30),
    latitude = (-60, 60),
    z = (-100, 0),
    halo = (4, 4, 4),
    partition = (1, 4),
    Δt = 100.0,
    stop_iteration = 4,
    output_interval = 2,
)

function lat_lon_mpi_script(config, filename)
    Nλ, Nφ, Nz = config.size
    lon1, lon2 = config.longitude
    lat1, lat2 = config.latitude
    z1, z2 = config.z
    Hλ, Hφ, Hz = config.halo
    px, py = config.partition

    return """
    using MPI
    MPI.Init()

    using Oceananigans
    using Oceananigans.DistributedComputations: Distributed, Partition
    using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface

    arch = Distributed(CPU(), partition=Partition($px, $py))

    grid = LatitudeLongitudeGrid(arch;
                                 size = ($Nλ, $Nφ, $Nz),
                                 longitude = ($lon1, $lon2),
                                 latitude = ($lat1, $lat2),
                                 z = ($z1, $z2),
                                 halo = ($Hλ, $Hφ, $Hz))

    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, free_surface=ExplicitFreeSurface())

    cᵢ(λ, φ, z) = sin(π * λ / 30) * cos(π * φ / 60) * (z + 100) / 100
    set!(model, c=cᵢ)

    simulation = Simulation(model; Δt=$(config.Δt), stop_iteration=$(config.stop_iteration))

    simulation.output_writers[:jld2] = JLD2Writer(model, model.tracers;
                                                  filename = "$filename",
                                                  schedule = IterationInterval($(config.output_interval)),
                                                  overwrite_existing = true,
                                                  with_halos = true)
    run!(simulation)

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
    """
end

function run_serial_lat_lon(config, filename)
    grid = LatitudeLongitudeGrid(CPU();
                                 size = config.size,
                                 longitude = config.longitude,
                                 latitude = config.latitude,
                                 z = config.z,
                                 halo = config.halo)

    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, free_surface=ExplicitFreeSurface())

    cᵢ(λ, φ, z) = sin(π * λ / 30) * cos(π * φ / 60) * (z + 100) / 100
    set!(model, c=cᵢ)

    simulation = Simulation(model; Δt=config.Δt, stop_iteration=config.stop_iteration)

    simulation.output_writers[:jld2] = JLD2Writer(model, model.tracers;
                                                  filename = filename,
                                                  schedule = IterationInterval(config.output_interval),
                                                  overwrite_existing = true,
                                                  with_halos = true)
    run!(simulation)
end

#####
##### TripolarGrid test configuration
#####

const tripolar_config = (
    size = (16, 16, 4),
    z = (-100, 0),
    halo = (4, 4, 4),
    north_poles_latitude = 55,
    first_pole_longitude = 70,
    partition = (1, 4),
    Δt = 100.0,
    stop_iteration = 4,
    output_interval = 2,
)

function tripolar_mpi_script(config, filename)
    Nλ, Nφ, Nz = config.size
    z1, z2 = config.z
    Hλ, Hφ, Hz = config.halo
    px, py = config.partition

    return """
    using MPI
    MPI.Init()

    using Oceananigans
    using Oceananigans.DistributedComputations: Distributed, Partition
    using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface
    using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

    arch = Distributed(CPU(), partition=Partition($px, $py))

    grid = TripolarGrid(arch;
                        size = ($Nλ, $Nφ, $Nz),
                        z = ($z1, $z2),
                        halo = ($Hλ, $Hφ, $Hz),
                        north_poles_latitude = $(config.north_poles_latitude),
                        first_pole_longitude = $(config.first_pole_longitude))

    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, free_surface=ExplicitFreeSurface())

    cᵢ(λ, φ, z) = cosd(φ) * (z + 100) / 100
    set!(model, c=cᵢ)

    simulation = Simulation(model; Δt=$(config.Δt), stop_iteration=$(config.stop_iteration))

    simulation.output_writers[:jld2] = JLD2Writer(model, model.tracers;
                                                  filename = "$filename",
                                                  schedule = IterationInterval($(config.output_interval)),
                                                  overwrite_existing = true,
                                                  with_halos = true)
    run!(simulation)

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
    """
end

function run_serial_tripolar(config, filename)
    grid = TripolarGrid(CPU();
                        size = config.size,
                        z = config.z,
                        halo = config.halo,
                        north_poles_latitude = config.north_poles_latitude,
                        first_pole_longitude = config.first_pole_longitude)

    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, free_surface=ExplicitFreeSurface())

    cᵢ(λ, φ, z) = cosd(φ) * (z + 100) / 100
    set!(model, c=cᵢ)

    simulation = Simulation(model; Δt=config.Δt, stop_iteration=config.stop_iteration)

    simulation.output_writers[:jld2] = JLD2Writer(model, model.tracers;
                                                  filename = filename,
                                                  schedule = IterationInterval(config.output_interval),
                                                  overwrite_existing = true,
                                                  with_halos = true)
    run!(simulation)
end

#####
##### Tests
#####

@testset "Distributed output combining - RectilinearGrid (2x2)" begin
    @info "Testing RectilinearGrid distributed output combining (2x2 partition)..."

    config = rectilinear_2x2_config
    dist_prefix = "rectilinear_2x2_distributed"
    serial_file = "rectilinear_2x2_serial.jld2"

    script = rectilinear_mpi_script(config, dist_prefix)
    run_mpi_script(script, "run_rectilinear_2x2.jl")

    run_serial_rectilinear(config, serial_file)

    test_combined_output_matches_serial(dist_prefix, serial_file, ["c", "u", "zflat"])

    @info "  RectilinearGrid (2x2) test passed!"

    rm(serial_file, force=true)
    cleanup_rank_files(dist_prefix)
end

@testset "Distributed output combining - RectilinearGrid slab (1x4)" begin
    @info "Testing RectilinearGrid slab decomposition (1x4) output combining..."

    config = rectilinear_slab_config
    dist_prefix = "rectilinear_slab_distributed"
    serial_file = "rectilinear_slab_serial.jld2"

    script = rectilinear_mpi_script(config, dist_prefix)
    run_mpi_script(script, "run_rectilinear_slab.jl")

    run_serial_rectilinear(config, serial_file)

    test_combined_output_matches_serial(dist_prefix, serial_file, ["c"])

    @info "  RectilinearGrid slab (1x4) test passed!"

    rm(serial_file, force=true)
    cleanup_rank_files(dist_prefix)
end

@testset "Distributed output combining - combine=false option" begin
    @info "Testing combine=false option..."

    config = rectilinear_2x2_config
    dist_prefix = "combine_false_test"

    script = rectilinear_mpi_script(config, dist_prefix)
    run_mpi_script(script, "run_combine_false.jl")

    # Load individual rank file with combine=false
    c_rank0 = FieldTimeSeries("$(dist_prefix)_rank0.jld2", "c"; combine=false)

    # The local field should have a smaller grid size (partitioned)
    Nx, Ny, Nz = config.size
    @test size(c_rank0.grid)[1] < Nx
    @test size(c_rank0.grid)[2] < Ny

    @info "  combine=false test passed!"

    # Keep files for OnDisk test
end

@testset "Distributed output combining - OnDisk backend" begin
    @info "Testing OnDisk backend with combined output..."

    config = rectilinear_2x2_config
    dist_prefix = "combine_false_test"  # Reuse from previous test
    serial_file = "ondisk_serial.jld2"

    # Load combined distributed output with OnDisk backend
    c_ondisk = FieldTimeSeries("$dist_prefix.jld2", "c"; backend=OnDisk())

    Nx, Ny, Nz = config.size
    @test size(c_ondisk.grid) == (Nx, Ny, Nz)

    run_serial_rectilinear(config, serial_file)

    c_serial = FieldTimeSeries(serial_file, "c"; backend=OnDisk())

    for n in 1:length(c_ondisk.times)
        @test interior(c_ondisk[n]) ≈ interior(c_serial[n])
    end

    @info "  OnDisk backend test passed!"

    rm(serial_file, force=true)
    cleanup_rank_files(dist_prefix)
end

@testset "Distributed output combining - LatitudeLongitudeGrid (1x4)" begin
    @info "Testing LatitudeLongitudeGrid distributed output combining (1x4 partition)..."

    config = lat_lon_config
    dist_prefix = "lat_lon_distributed"
    serial_file = "lat_lon_serial.jld2"

    script = lat_lon_mpi_script(config, dist_prefix)
    run_mpi_script(script, "run_lat_lon.jl")

    run_serial_lat_lon(config, serial_file)

    test_combined_output_matches_serial(dist_prefix, serial_file, ["c"])

    @info "  LatitudeLongitudeGrid (1x4) test passed!"

    rm(serial_file, force=true)
    cleanup_rank_files(dist_prefix)
end

@testset "Distributed output combining - TripolarGrid (1x4)" begin
    @info "Testing TripolarGrid distributed output combining (1x4 partition)..."

    config = tripolar_config
    dist_prefix = "tripolar_distributed"
    serial_file = "tripolar_serial.jld2"

    script = tripolar_mpi_script(config, dist_prefix)
    run_mpi_script(script, "run_tripolar.jl")

    run_serial_tripolar(config, serial_file)

    test_combined_output_matches_serial(dist_prefix, serial_file, ["c"])

    @info "  TripolarGrid (1x4) test passed!"

    rm(serial_file, force=true)
    cleanup_rank_files(dist_prefix)
end
