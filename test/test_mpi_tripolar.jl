include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

using MPI

fold_topologies = (RightCenterFolded, RightFaceFolded)

tripolar_reconstructed_grid_script(fold_topology) = """
    using MPI
    MPI.Init()
    using Test

    include("distributed_tests_utils.jl")

    archs = [Distributed(CPU(), partition=Partition(1, 4)),
             Distributed(CPU(), partition=Partition(2, 2))]

    for arch in archs
        local_grid  = TripolarGrid(arch; size = (12, 20, 1), z = (-1000, 0), halo = (2, 2, 2), fold_topology = $fold_topology)
        global_grid = TripolarGrid(size = (12, 20, 1), z = (-1000, 0), halo = (2, 2, 2), fold_topology = $fold_topology)

        reconstruct_grid = reconstruct_global_grid(local_grid)

        @test reconstruct_grid == global_grid

        nx, ny, _ = size(local_grid)
        rx, ry, _ = arch.local_index .- 1

        jrange = 1 + ry * ny : (ry + 1) * ny
        irange = 1 + rx * nx : (rx + 1) * nx

        for var in [:Δxᶠᶠᵃ, :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ,
                    :Δyᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ,
                    :Azᶠᶠᵃ, :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ]

            @test getproperty(local_grid, var)[1:nx, 1:ny] == getproperty(global_grid, var)[irange, jrange]
            @test getproperty(local_grid, var)[1:nx, 1:ny] == getproperty(global_grid, var)[irange, jrange]
            @test getproperty(local_grid, var)[1:nx, 1:ny] == getproperty(global_grid, var)[irange, jrange]
        end
    end
"""

tripolar_reconstructed_field_script(fold_topology) = """
    using MPI
    MPI.Init()
    using Test

    include("distributed_tests_utils.jl")

    archs = [Distributed(CPU(), partition=Partition(1, 4)),
             Distributed(CPU(), partition=Partition(2, 2))]

    u = [i + 10 * j for i in 1:40, j in 1:40]
    v = [i + 10 * j for i in 1:40, j in 1:$(fold_topology == RightCenterFolded ? 40 : 41)]
    c = [i + 10 * j for i in 1:40, j in 1:40]

    for arch in archs
        local_grid = TripolarGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5), fold_topology = $fold_topology)

        up = XFaceField(local_grid)
        vp = YFaceField(local_grid)
        cp = CenterField(local_grid)

        set!(up, u)
        set!(vp, v)
        set!(cp, c)

        global_grid = TripolarGrid(size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5), fold_topology = $fold_topology)

        us = XFaceField(global_grid)
        vs = YFaceField(global_grid)
        cs = CenterField(global_grid)

        set!(us, u)
        set!(vs, v)
        set!(cs, c)

        @test us == reconstruct_global_field(up)
        @test vs == reconstruct_global_field(vp)
        @test cs == reconstruct_global_field(cp)
    end
"""

@testset "Test distributed TripolarGrid $fold_topology..." for fold_topology in fold_topologies
    write("distributed_tripolar_grid_$fold_topology.jl", tripolar_reconstructed_grid_script(fold_topology))
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 $("distributed_tripolar_grid_$fold_topology.jl")`)
    rm("distributed_tripolar_grid_$fold_topology.jl")

    write("distributed_tripolar_field_$fold_topology.jl", tripolar_reconstructed_field_script(fold_topology))
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 $("distributed_tripolar_field_$fold_topology.jl")`)
    rm("distributed_tripolar_field_$fold_topology.jl")
end

tripolar_boundary_conditions_script(fold_topology) = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (20, 20, 1), z = (-1000, 0), fold_topology = $fold_topology)

    # Build initial condition
    serial_grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0), fold_topology = $fold_topology)

    vs =  YFaceField(serial_grid)
    cs = CenterField(serial_grid)

    I_v = [i + j * 100 for i in 1:20, j in 1:$(fold_topology == RightCenterFolded ? 20 : 21)]
    I_c = [i + j * 100 for i in 1:20, j in 1:20]

    set!(vs, I_v)
    set!(cs, I_c)

    fill_halo_regions!((vs, cs))

    v =  YFaceField(grid)
    c = CenterField(grid)

    set!(v, vs)
    set!(c, cs)

    fill_halo_regions!((v, c))
    filename = "distributed_$(fold_topology)_boundary_conditions_" * string(arch.local_rank) * ".jld2"

    jldopen(filename, "w") do file
        file["v"] = v.data
        file["c"] = c.data
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

@testset "Test distributed TripolarGrid boundary conditions $fold_topology..." for fold_topology in fold_topologies
    # Run the serial computation
    grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0); fold_topology)

    v = YFaceField(grid)
    c = CenterField(grid)

    Nyv = fold_topology == RightCenterFolded ? 20 : 21
    I_v = [i + j * 100 for i in 1:20, j in 1:Nyv]
    I_c = [i + j * 100 for i in 1:20, j in 1:20]

    set!(v, I_v)
    set!(c, I_c)

    fill_halo_regions!((v, c))

    write("distributed_$(fold_topology)_boundary_tests.jl", tripolar_boundary_conditions_script(fold_topology))
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 $("distributed_$(fold_topology)_boundary_tests.jl")`)
    rm("distributed_$(fold_topology)_boundary_tests.jl")

    # Retrieve Parallel quantities from rank 1 (the north-west rank)
    vp1 = jldopen("distributed_$(fold_topology)_boundary_conditions_1.jld2")["v"];
    cp1 = jldopen("distributed_$(fold_topology)_boundary_conditions_1.jld2")["c"];

    # Retrieve Parallel quantities from rank 3 (the north-east rank)
    vp3 = jldopen("distributed_$(fold_topology)_boundary_conditions_3.jld2")["v"];
    cp3 = jldopen("distributed_$(fold_topology)_boundary_conditions_3.jld2")["c"];

    @test v.data[-3:14, end-3:end-1, 1] ≈ vp1.parent[:, end-3:end-1, 5]
    @test c.data[-3:14, end-3:end-1, 1] ≈ cp1.parent[:, end-3:end-1, 5]
    @test v.data[7:end, 7:end-1, 1] ≈ vp3.parent[:, 1:end-1, 5]
    @test c.data[7:end, 7:end-1, 1] ≈ cp3.parent[:, 1:end-1, 5]

    for rank in 0:3
        rm("distributed_$(fold_topology)_boundary_conditions_$(rank).jld2", force=true)
    end
end

run_slab_distributed_grid(fold_topology) = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_$(fold_topology)_yslab_tripolar.jld2"; fold_topology = $fold_topology)
"""

run_pencil_distributed_grid(fold_topology) = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_$(fold_topology)_pencil_tripolar.jld2"; fold_topology = $fold_topology)
"""

run_large_pencil_distributed_grid(fold_topology) = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(4, 2))
    run_distributed_tripolar_grid(arch, "distributed_$(fold_topology)_large_pencil_tripolar.jld2"; fold_topology = $fold_topology)
"""

@testset "Test distributed TripolarGrid simulations $fold_topology..." for fold_topology in fold_topologies
    # Run the serial computation
    grid  = TripolarGrid(size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5); fold_topology)
    grid  = analytical_immersed_tripolar_grid(grid)
    model = run_distributed_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = model.velocities
    cs = model.tracers.c
    ηs = model.free_surface.displacement

    us = interior(us, :, :, 1)
    vs = interior(vs, :, :, 1)
    cs = interior(cs, :, :, 1)

    # Run the distributed grid simulation with a slab configuration
    write("distributed_slab_tests.jl", run_slab_distributed_grid(fold_topology))
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 distributed_slab_tests.jl`)
    rm("distributed_slab_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_$(fold_topology)_yslab_tripolar.jld2")["u"]
    vp = jldopen("distributed_$(fold_topology)_yslab_tripolar.jld2")["v"]
    cp = jldopen("distributed_$(fold_topology)_yslab_tripolar.jld2")["c"]
    ηp = jldopen("distributed_$(fold_topology)_yslab_tripolar.jld2")["η"]

    rm("distributed_$(fold_topology)_yslab_tripolar.jld2")

    # Test slab partitioning
    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)

    # Run the distributed grid simulation with a pencil configuration
    write("distributed_pencil_tests.jl", run_pencil_distributed_grid(fold_topology))
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 distributed_pencil_tests.jl`)
    rm("distributed_pencil_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_$(fold_topology)_pencil_tripolar.jld2")["u"]
    vp = jldopen("distributed_$(fold_topology)_pencil_tripolar.jld2")["v"]
    ηp = jldopen("distributed_$(fold_topology)_pencil_tripolar.jld2")["η"]
    cp = jldopen("distributed_$(fold_topology)_pencil_tripolar.jld2")["c"]

    rm("distributed_$(fold_topology)_pencil_tripolar.jld2")

    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)

    # We try now with more ranks in the x-direction. This is not a trivial
    # test as we are now splitting, not only where the singularities are, but
    # also in the middle of the north fold. This is a more challenging test
    write("distributed_large_pencil_tests.jl", run_large_pencil_distributed_grid(fold_topology))
    run(`$(mpiexec()) -n 8 $(Base.julia_cmd()) -O0 distributed_large_pencil_tests.jl`)
    rm("distributed_large_pencil_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_$(fold_topology)_large_pencil_tripolar.jld2")["u"]
    vp = jldopen("distributed_$(fold_topology)_large_pencil_tripolar.jld2")["v"]
    ηp = jldopen("distributed_$(fold_topology)_large_pencil_tripolar.jld2")["η"]
    cp = jldopen("distributed_$(fold_topology)_large_pencil_tripolar.jld2")["c"]

    rm("distributed_$(fold_topology)_large_pencil_tripolar.jld2")

    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)
end

tripolar_compute_output_writer_script(fold_topology) = """
    using MPI
    MPI.Init()
    using Test

    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5), fold_topology = $fold_topology)
    grid = analytical_immersed_tripolar_grid(grid)

    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                        tracers = :c,
                                        tracer_advection = WENO(),
                                        momentum_advection = WENOVectorInvariant(order=3),
                                        coriolis = HydrostaticSphericalCoriolis())

    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)
    set!(model, c=ηᵢ, η=ηᵢ)

    computed_surface = Field(model.tracers.c + 1; indices=(:, :, 1))

    simulation = Simulation(model, Δt=5minutes, stop_iteration=1)
    filename = "distributed_tripolar_compute_output_writer.jld2"

    simulation.output_writers[:computed_surface] = JLD2Writer(model, (; computed_surface);
                                                              filename,
                                                              schedule = IterationInterval(1),
                                                              with_halos = false,
                                                              overwrite_existing = true)

    run!(simulation)

    @root begin
        output = jldopen(filename, "r") do file
            file["timeseries/computed_surface/0"]
        end

        @test size(output) == (40, 40, 1)
        rm(filename, force=true)
    end
    MPI.Finalize()
"""

@testset "Test distributed TripolarGrid compute-output writer regression $fold_topology..." for fold_topology in fold_topologies
    script = "distributed_tripolar_compute_writer_tests.jl"
    write(script, tripolar_compute_output_writer_script(fold_topology))
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) -O0 $script`)
    rm(script)
end
