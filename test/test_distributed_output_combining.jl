include("dependencies_for_runtests.jl")

using JLD2

# Distributed output combining tests
#
# These tests verify that distributed output (with _rank0, _rank1, etc. suffixes)
# can be automatically combined into a global FieldTimeSeries that matches
# output from an equivalent non-distributed simulation.

#####
##### Test scripts to be run with MPI
#####

# Script that runs a distributed simulation and saves output
distributed_simulation_script = """
    using MPI
    MPI.Init()
    
    using Oceananigans
    using Oceananigans.DistributedComputations: Distributed, Partition
    
    # Create distributed architecture with 2x2 decomposition
    arch = Distributed(CPU(), partition=Partition(2, 2))
    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    
    # Global grid parameters
    Nx, Ny, Nz = 8, 8, 4
    Lx, Ly, Lz = 1.0, 1.0, 0.5
    
    # Create distributed grid
    distributed_grid = RectilinearGrid(arch;
                                        topology = (Periodic, Periodic, Bounded),
                                        size = (Nx, Ny, Nz),
                                        extent = (Lx, Ly, Lz))
    
    # Create distributed model
    distributed_model = NonhydrostaticModel(; grid=distributed_grid, tracers=:c)
    
    # Set initial conditions
    cᵢ(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly) * (z + Lz) / Lz
    uᵢ(x, y, z) = 0.1 * sin(2π * x / Lx)
    set!(distributed_model, c=cᵢ, u=uᵢ)
    
    simulation = Simulation(distributed_model; Δt=1.0, stop_iteration=10)
    
    outputs = merge(distributed_model.velocities, distributed_model.tracers)
    
    simulation.output_writers[:jld2] = JLD2Writer(distributed_model, outputs;
                                                   filename = "distributed_output_test",
                                                   schedule = IterationInterval(5),
                                                   overwrite_existing = true,
                                                   with_halos = true)
    
    run!(simulation)
    
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

# Script that runs slab decomposition (1x4)
slab_simulation_script = """
    using MPI
    MPI.Init()
    
    using Oceananigans
    using Oceananigans.DistributedComputations: Distributed, Partition
    
    arch = Distributed(CPU(), partition=Partition(1, 4))
    
    Nx, Ny, Nz = 8, 16, 4
    Lx, Ly, Lz = 1.0, 2.0, 0.5
    
    distributed_grid = RectilinearGrid(arch;
                                        topology = (Periodic, Periodic, Bounded),
                                        size = (Nx, Ny, Nz),
                                        extent = (Lx, Ly, Lz))
    
    distributed_model = NonhydrostaticModel(; grid=distributed_grid, tracers=:c)
    cᵢ(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
    set!(distributed_model, c=cᵢ)
    
    simulation = Simulation(distributed_model; Δt=1.0, stop_iteration=4)
    
    simulation.output_writers[:jld2] = JLD2Writer(distributed_model, distributed_model.tracers;
                                                   filename = "slab_output_test",
                                                   schedule = IterationInterval(2),
                                                   overwrite_existing = true,
                                                   with_halos = true)
    
    run!(simulation)
    
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

#####
##### Tests
#####

@testset "Distributed output combining - RectilinearGrid (2x2)" begin
    @info "Testing RectilinearGrid distributed output combining (2x2 partition)..."
    
    # Run distributed simulation
    write("run_distributed_output.jl", distributed_simulation_script)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 run_distributed_output.jl`)
    rm("run_distributed_output.jl")
    
    # Run equivalent serial simulation
    Nx, Ny, Nz = 8, 8, 4
    Lx, Ly, Lz = 1.0, 1.0, 0.5
    
    serial_grid = RectilinearGrid(CPU();
                                   topology = (Periodic, Periodic, Bounded),
                                   size = (Nx, Ny, Nz),
                                   extent = (Lx, Ly, Lz))
    
    serial_model = NonhydrostaticModel(; grid=serial_grid, tracers=:c)
    cᵢ(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly) * (z + Lz) / Lz
    uᵢ(x, y, z) = 0.1 * sin(2π * x / Lx)
    set!(serial_model, c=cᵢ, u=uᵢ)
    
    simulation = Simulation(serial_model; Δt=1.0, stop_iteration=10)
    
    simulation.output_writers[:jld2] = JLD2Writer(serial_model, 
                                                   merge(serial_model.velocities, serial_model.tracers);
                                                   filename = "serial_output_test.jld2",
                                                   schedule = IterationInterval(5),
                                                   overwrite_existing = true,
                                                   with_halos = true)
    run!(simulation)
    
    # Load combined distributed output (should automatically detect rank files)
    c_distributed = FieldTimeSeries("distributed_output_test.jld2", "c")
    u_distributed = FieldTimeSeries("distributed_output_test.jld2", "u")
    
    # Load serial output
    c_serial = FieldTimeSeries("serial_output_test.jld2", "c")
    u_serial = FieldTimeSeries("serial_output_test.jld2", "u")
    
    # Check grid sizes match
    @test size(c_distributed.grid) == size(c_serial.grid)
    @test size(u_distributed.grid) == size(u_serial.grid)
    
    # Check time series sizes match
    @test size(c_distributed) == size(c_serial)
    @test size(u_distributed) == size(u_serial)
    
    # Check times match
    @test c_distributed.times ≈ c_serial.times
    
    # Check data matches for each time step
    Nt = length(c_distributed.times)
    for n in 1:Nt
        @test interior(c_distributed[n]) ≈ interior(c_serial[n])
        @test interior(u_distributed[n]) ≈ interior(u_serial[n])
    end
    
    @info "  RectilinearGrid (2x2) test passed! ✓"
    
    # Clean up
    rm("serial_output_test.jld2", force=true)
    for r in 0:3
        rm("distributed_output_test_rank$r.jld2", force=true)
    end
end

@testset "Distributed output combining - Slab decomposition (1x4)" begin
    @info "Testing slab decomposition (1x4) output combining..."
    
    # Run distributed simulation
    write("run_slab_output.jl", slab_simulation_script)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 run_slab_output.jl`)
    rm("run_slab_output.jl")
    
    # Run equivalent serial simulation
    Nx, Ny, Nz = 8, 16, 4
    Lx, Ly, Lz = 1.0, 2.0, 0.5
    
    serial_grid = RectilinearGrid(CPU();
                                   topology = (Periodic, Periodic, Bounded),
                                   size = (Nx, Ny, Nz),
                                   extent = (Lx, Ly, Lz))
    
    serial_model = NonhydrostaticModel(; grid=serial_grid, tracers=:c)
    cᵢ(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
    set!(serial_model, c=cᵢ)
    
    simulation = Simulation(serial_model; Δt=1.0, stop_iteration=4)
    
    simulation.output_writers[:jld2] = JLD2Writer(serial_model, serial_model.tracers;
                                                   filename = "slab_serial_test.jld2",
                                                   schedule = IterationInterval(2),
                                                   overwrite_existing = true,
                                                   with_halos = true)
    run!(simulation)
    
    # Load and compare
    c_distributed = FieldTimeSeries("slab_output_test.jld2", "c")
    c_serial = FieldTimeSeries("slab_serial_test.jld2", "c")
    
    @test size(c_distributed) == size(c_serial)
    @test c_distributed.times ≈ c_serial.times
    
    for n in 1:length(c_distributed.times)
        @test interior(c_distributed[n]) ≈ interior(c_serial[n])
    end
    
    @info "  Slab decomposition (1x4) test passed! ✓"
    
    # Clean up
    rm("slab_serial_test.jld2", force=true)
    for r in 0:3
        rm("slab_output_test_rank$r.jld2", force=true)
    end
end

@testset "Distributed output combining - combine=false option" begin
    @info "Testing combine=false option..."
    
    # The rank files should still exist from the previous test, 
    # but let's create fresh ones
    write("run_distributed_output.jl", distributed_simulation_script)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 run_distributed_output.jl`)
    rm("run_distributed_output.jl")
    
    # Load individual rank file with combine=false
    c_rank0 = FieldTimeSeries("distributed_output_test_rank0.jld2", "c"; combine=false)
    
    # The local field should have a smaller grid size (partitioned)
    @test size(c_rank0.grid)[1] < 8  # Should be 4 for 2x2 partition
    @test size(c_rank0.grid)[2] < 8
    
    @info "  combine=false test passed! ✓"
    
    # Don't clean up yet - use for OnDisk test
end

@testset "Distributed output combining - OnDisk backend" begin
    @info "Testing OnDisk backend with combined output..."
    
    # Rank files should exist from the previous test
    # Load combined distributed output with OnDisk backend
    c_ondisk = FieldTimeSeries("distributed_output_test.jld2", "c"; backend=OnDisk())
    
    # Check grid size is global (8x8x4)
    @test size(c_ondisk.grid) == (8, 8, 4)
    
    # Check we can index and get correct data
    @test length(c_ondisk.times) == 3  # iterations 0, 5, 10
    
    # Load serial output for comparison
    Nx, Ny, Nz = 8, 8, 4
    Lx, Ly, Lz = 1.0, 1.0, 0.5
    
    serial_grid = RectilinearGrid(CPU();
                                   topology = (Periodic, Periodic, Bounded),
                                   size = (Nx, Ny, Nz),
                                   extent = (Lx, Ly, Lz))
    
    serial_model = NonhydrostaticModel(; grid=serial_grid, tracers=:c)
    cᵢ(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly) * (z + Lz) / Lz
    uᵢ(x, y, z) = 0.1 * sin(2π * x / Lx)
    set!(serial_model, c=cᵢ, u=uᵢ)
    
    simulation = Simulation(serial_model; Δt=1.0, stop_iteration=10)
    
    simulation.output_writers[:jld2] = JLD2Writer(serial_model, 
                                                   merge(serial_model.velocities, serial_model.tracers);
                                                   filename = "ondisk_serial_test.jld2",
                                                   schedule = IterationInterval(5),
                                                   overwrite_existing = true,
                                                   with_halos = true)
    run!(simulation)
    
    # Load serial with OnDisk for fair comparison
    c_serial_ondisk = FieldTimeSeries("ondisk_serial_test.jld2", "c"; backend=OnDisk())
    
    # Compare each time step
    for n in 1:length(c_ondisk.times)
        @test interior(c_ondisk[n]) ≈ interior(c_serial_ondisk[n])
    end
    
    @info "  OnDisk backend test passed! ✓"
    
    # Clean up
    rm("ondisk_serial_test.jld2", force=true)
    for r in 0:3
        rm("distributed_output_test_rank$r.jld2", force=true)
    end
end
