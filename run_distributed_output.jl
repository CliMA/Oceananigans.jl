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
