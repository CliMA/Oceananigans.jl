include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

using MPI

tripolar_reconstructed_grid = """
    using MPI
    MPI.Init()
    using Test

    include("distributed_tests_utils.jl")

    archs = [Distributed(CPU(), partition=Partition(1, 4)),
             Distributed(CPU(), partition=Partition(2, 2))]

    for arch in archs
        local_grid  = TripolarGrid(arch; size = (12, 20, 1), z = (-1000, 0), halo = (2, 2, 2))
        global_grid = TripolarGrid(size = (12, 20, 1), z = (-1000, 0), halo = (2, 2, 2))

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

tripolar_reconstructed_field = """
    using MPI
    MPI.Init()
    using Test

    include("distributed_tests_utils.jl")

    archs = [Distributed(CPU(), partition=Partition(1, 4)),
             Distributed(CPU(), partition=Partition(2, 2))]

    u = [i + 10 * j for i in 1:40, j in 1:40]
    v = [i + 10 * j for i in 1:40, j in 1:40]
    c = [i + 10 * j for i in 1:40, j in 1:40]

    for arch in archs
        local_grid = TripolarGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))

        up = XFaceField(local_grid)
        vp = YFaceField(local_grid)
        cp = CenterField(local_grid)

        set!(up, u)
        set!(vp, v)
        set!(cp, c)

        global_grid = TripolarGrid(size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))

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

@testset "Test distributed TripolarGrid..." begin
    write("distributed_tripolar_grid.jl", tripolar_reconstructed_grid)
    run(`$(mpiexec()) -n 4 julia --project -O0 distributed_tripolar_grid.jl`)
    rm("distributed_tripolar_grid.jl")

    write("distributed_tripolar_field.jl", tripolar_reconstructed_field)
    run(`$(mpiexec()) -n 4 julia --project -O0 distributed_tripolar_field.jl`)
    rm("distributed_tripolar_field.jl")
end

tripolar_boundary_conditions = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (20, 20, 1), z = (-1000, 0))

    # Build initial condition
    serial_grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))

    vs =  YFaceField(serial_grid)
    cs = CenterField(serial_grid)

    I1 = [i + j * 100 for i in 1:20, j in 1:20]

    set!(vs, I1)
    set!(cs, I1)

    fill_halo_regions!((vs, cs))

    v =  YFaceField(grid)
    c = CenterField(grid)

    set!(v, vs)
    set!(c, cs)

    fill_halo_regions!((v, c))
    filename = "distributed_tripolar_boundary_conditions_" * string(arch.local_rank) * ".jld2"

    jldopen(filename, "w") do file
        file["v"] = v.data
        file["c"] = c.data
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
"""

@testset "Test distributed TripolarGrid boundary conditions..." begin
    # Run the serial computation
    grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))

    I1 = [i + j * 100 for i in 1:20, j in 1:20]

    v = YFaceField(grid)
    c = CenterField(grid)

    set!(v, I1)
    set!(c, I1)

    fill_halo_regions!((v, c))

    write("distributed_boundary_tests.jl", tripolar_boundary_conditions)
    run(`$(mpiexec()) -n 4 julia --project -O0 distributed_boundary_tests.jl`)
    rm("distributed_boundary_tests.jl")

    # Retrieve Parallel quantities from rank 1 (the north-west rank)
    vp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["v"];
    cp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["c"];

    # Retrieve Parallel quantities from rank 3 (the north-east rank)
    vp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["v"];
    cp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["c"];

    @test v.data[-3:14, end-3:end-1, 1] ≈ vp1.parent[:, end-3:end-1, 5]
    @test c.data[-3:14, end-3:end-1, 1] ≈ cp1.parent[:, end-3:end-1, 5]
    @test v.data[7:end, 7:end-1, 1] ≈ vp3.parent[:, 1:end-1, 5]
    @test c.data[7:end, 7:end-1, 1] ≈ cp3.parent[:, 1:end-1, 5]
end

run_slab_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_yslab_tripolar.jld2")
"""

run_pencil_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_pencil_tripolar.jld2")
"""

run_large_pencil_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(4, 2))
    run_distributed_tripolar_grid(arch, "distributed_large_pencil_tripolar.jld2")
"""

@testset "Test distributed TripolarGrid simulations..." begin
    # Run the serial computation
    grid  = TripolarGrid(size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))
    grid  = analytical_immersed_tripolar_grid(grid)
    model = run_distributed_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = model.velocities
    cs = model.tracers.c
    ηs = model.free_surface.η

    us = interior(us, :, :, 1)
    vs = interior(vs, :, :, 1)
    cs = interior(cs, :, :, 1)
    # Run the distributed grid simulation with a slab configuration
    write("distributed_slab_tests.jl", run_slab_distributed_grid)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 distributed_slab_tests.jl`)
    rm("distributed_slab_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_yslab_tripolar.jld2")["u"]
    vp = jldopen("distributed_yslab_tripolar.jld2")["v"]
    cp = jldopen("distributed_yslab_tripolar.jld2")["c"]
    ηp = jldopen("distributed_yslab_tripolar.jld2")["η"]

    rm("distributed_yslab_tripolar.jld2")

    # Test slab partitioning
    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)

    # Run the distributed grid simulation with a pencil configuration
    write("distributed_tests.jl", run_pencil_distributed_grid)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 distributed_tests.jl`)
    rm("distributed_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_pencil_tripolar.jld2")["u"]
    vp = jldopen("distributed_pencil_tripolar.jld2")["v"]
    ηp = jldopen("distributed_pencil_tripolar.jld2")["η"]
    cp = jldopen("distributed_pencil_tripolar.jld2")["c"]

    rm("distributed_pencil_tripolar.jld2")

    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)

    # We try now with more ranks in the x-direction. This is not a trivial
    # test as we are now splitting, not only where the singularities are, but
    # also in the middle of the north fold. This is a more challenging test
    write("distributed_large_pencil_tests.jl", run_large_pencil_distributed_grid)
    run(`$(mpiexec()) -n 8 $(Base.julia_cmd()) --project -O0 distributed_large_pencil_tests.jl`)
    rm("distributed_large_pencil_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_large_pencil_tripolar.jld2")["u"]
    vp = jldopen("distributed_large_pencil_tripolar.jld2")["v"]
    ηp = jldopen("distributed_large_pencil_tripolar.jld2")["η"]
    cp = jldopen("distributed_large_pencil_tripolar.jld2")["c"]

    rm("distributed_large_pencil_tripolar.jld2")

    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)
end