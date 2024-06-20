using MPI

MPI.Init()

include("dependencies_for_runtests.jl")

using Oceananigans.DistributedComputations: 
                ParallelFields, 
                transpose_z_to_y!, 
                transpose_y_to_z!,
                transpose_y_to_x!,
                transpose_x_to_y!

function test_transpose(grid_points, ranks, topo, child_arch)
    arch = Distributed(child_arch, partition=Partition(ranks...))
    grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    loc = (Center, Center, Center)
    ϕ = Field(loc, grid, ComplexF64)
    Φ = ParallelFields(ϕ)

    # Fill ϕ with random data
    set!(ϕ, (x, y, z) ->  rand(ComplexF64))
    set!(Φ.zfield, ϕ)
    
    # Complete a full transposition cycle
    transpose_z_to_y!(Φ)
    transpose_y_to_x!(Φ)
    transpose_x_to_y!(Φ)
    transpose_y_to_z!(Φ)

    # Check that the data is unchanged
    same_real_part = all(real.(Array(interior(ϕ))) .== real.(Array(interior(Φ.zfield))))
    same_imag_part = all(imag.(Array(interior(ϕ))) .== imag.(Array(interior(Φ.zfield))))

    return same_real_part & same_imag_part
end

@testset "Distributed Transpose" begin
    child_arch = test_child_arch()

    for topology in ((Periodic, Periodic, Periodic), 
                     (Periodic, Periodic, Bounded),
                     (Periodic, Bounded, Bounded),
                     (Bounded, Bounded, Bounded))
        @info "  Testing 3D transpose with topology $topology..."
        @test test_transpose((44, 44, 8), (4, 1, 1), topology, child_arch)
        @test test_transpose((16, 44, 8), (4, 1, 1), topology, child_arch)
        @test test_transpose((44, 44, 8), (1, 4, 1), topology, child_arch)
        @test test_transpose((44, 16, 8), (1, 4, 1), topology, child_arch)
        @test test_transpose((16, 44, 8), (1, 4, 1), topology, child_arch)
        @test test_transpose((44, 16, 8), (2, 2, 1), topology, child_arch)
        @test test_transpose((16, 44, 8), (2, 2, 1), topology, child_arch)
    end
end

