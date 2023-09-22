using MPI

MPI.Init()

include("dependencies_for_runtests.jl")

using Oceananigans.DistributedComputations: 
                ParallelFields, 
                transpose_z_to_y!, 
                transpose_y_to_z!,
                transpose_y_to_x!,
                transpose_x_to_y!

function test_transpose(grid_points, ranks, topo)
    arch = Distributed(CPU(), ranks=ranks, topology=topo)
    grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    ϕ = CenterField(grid)
    Φ = ParallelFields(ϕ)

    # Fill ϕ with random data
    set!(ϕ, (x, y, z) -> rand())
    set!(Φ.zfield, ϕ)
    
    # Complete a full transposition cycle
    transpose_z_to_y!(Φ)
    transpose_y_to_x!(Φ)
    transpose_x_to_y!(Φ)
    transpose_y_to_z!(Φ)

    # Check that the data is unchanged
    return all(interior(ϕ) .== interior(Φ.zfield))
end



@testset "Distributed Transpose" begin
    for topology in ((Periodic, Periodic, Periodic), 
                     (Periodic, Periodic, Bounded),
                     (Periodic, Bounded, Bounded),
                     (Bounded, Bounded, Bounded))
        @info "  Testing 3D transpose with topology $topology..."
        @test test_transpose((11, 44, 8), (4, 1, 1), topology)
        @test test_transpose((4,  44, 8), (4, 1, 1), topology)
        @test test_transpose((44, 11, 8), (1, 4, 1), topology)
        @test test_transpose((44,  4, 8), (1, 4, 1), topology)
        @test test_transpose((16, 11, 8), (1, 4, 1), topology)
        @test test_transpose((22,  8, 8), (2, 2, 1), topology)
        @test test_transpose(( 8, 22, 8), (2, 2, 1), topology)
    end
end

