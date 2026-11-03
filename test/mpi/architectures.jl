include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using MPI
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: cpu_architecture, child_architecture, partition, reconstruct_global_grid, check_field_tags
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Operators: Azᶜᶜᶜ, Δxᶠᶜᶜ
using CUDA

@testset "Distributed macros" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    @onrank 0 begin
        @test rank == 0
    end

    @root begin
        @test rank == 0
    end

    @onrank 1 begin
        @test rank == 1
    end

    @onrank 2 begin
        @test rank == 2
    end

    @onrank 3 begin
        @test rank == 3
    end

    a = Int[]

    @distribute for i in 1:10
        push!(a, i)
    end

    @root begin
        @test a == [1, 5, 9]
    end

    @onrank 1 begin
        @test a == [2, 6, 10]
    end

    @onrank 2 begin
        @test a == [3, 7]
    end

    @onrank 3 begin
        @test a == [4, 8]
    end

    split_comm = MPI.Comm_split(MPI.COMM_WORLD, rank % 2, rank)

    a = Int[]

    @distribute split_comm for i in 1:10
        push!(a, i)
    end

    @onrank split_comm 0 @test a == [1, 3, 5, 7, 9]
    @onrank split_comm 1 @test a == [2, 4, 6, 8, 10]
end

@testset "Distributed grids have the nodes of the global grid" begin
    kw = (size = (140, 92, 2), longitude = (-76 + 2/12, -64 - 2/12), latitude = (34 + 2/12, 42 - 2/12), z = (-10, 0), halo = (3, 3, 3))
    grid = LatitudeLongitudeGrid(CPU(); kw...)

    for arch in archs
        local_grid = LatitudeLongitudeGrid(arch; kw...)
        Nx, Ny, _ = size(local_grid)

        for f in ((i, j, g) -> λnode(i, j, 1, g, Face(), Center(), Center()),
                  (i, j, g) -> φnode(i, j, 1, g, Center(), Center(), Center()),
                  (i, j, g) -> Azᶜᶜᶜ(i, j, 1, g),
                  (i, j, g) -> Δxᶠᶜᶜ(i, j, 1, g))

            global_values = [f(i, j, grid) for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:1]
            local_values = [f(i, j, local_grid) for i in 1:Nx, j in 1:Ny, k in 1:1]
            @test local_values == partition(global_values, cpu_architecture(arch), size(local_values))
        end
    end
end

@testset "Reconstructed distributed grids equal the global grid" begin
    topologies = ((Periodic, Periodic, Bounded),
                  (Periodic, Bounded, Bounded),
                  (Bounded, Periodic, Bounded),
                  (Bounded, Bounded, Bounded))

    for arch in archs, topology in topologies
        child_arch = child_architecture(arch)

        kw = (size = (100, 100, 2), extent = (1, 2, 3), topology = topology)
        local_grid = RectilinearGrid(arch; kw...)
        global_grid = RectilinearGrid(child_arch; kw...)
        @test reconstruct_global_grid(local_grid) == global_grid

        # LatitudeLongitudeGrid cannot be Periodic in latitude
        if topology[2] == Bounded
            kw = (size = (100, 100, 2), longitude = (-20, 20), latitude = (-10, 10), z = (-1000, 0), topology = topology)
            local_grid = LatitudeLongitudeGrid(arch; kw...)
            global_grid = LatitudeLongitudeGrid(child_arch; kw...)
            @test reconstruct_global_grid(local_grid) == global_grid
        end
    end
end

@testset "Indexing a FieldTimeSeries on some ranks keeps halo tags matched" begin
    f(x, y, z) = sin(2π * x) * cos(π * y)

    for arch in archs
        grid = RectilinearGrid(arch; size = (100, 100, 2), extent = (1, 2, 3))
        fts = FieldTimeSeries{Center, Center, Center}(grid, 0:3)

        # eg only the ranks that own an open boundary read its data
        arch.local_rank == 0 && fts[2]
        @test isnothing(check_field_tags(arch))
        @test fts[1].communication_buffers === fts[1].communication_buffers

        for n in 1:4
            set!(fts[n], f)
        end
        fill_halo_regions!(fts)

        c = CenterField(grid)
        set!(c, f)
        fill_halo_regions!(c)
        @test Array(parent(fts[2])) == Array(parent(c))
    end
end

#=
@testset "Distributed architectures" begin
    for arch in test_architectures()
        child_arch = child_architecture(arch)

        communicator = MPI.COMM_WORLD

        if child_arch isa Oceananigans.Architectures.GPU
            # Check that no device is the same!
            local_comm = MPI.Comm_split_type(communicator, MPI.COMM_TYPE_SHARED, arch.local_rank)
            node_rank  = MPI.Comm_rank(local_comm)
            device_number = CUDA.device().handle
            # We are testing on the same node, therefore we can
            # assume the GPU number changes with the rank
            @test node_rank == device_number
        end
    end
end
=#
