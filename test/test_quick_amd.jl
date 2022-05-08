include("dependencies_for_runtests.jl")
using GPUArrays

#####
##### Run some basic AMD tests
##### - this test file is meant to be used for early development for
##### - quick CI using amdgpu buildkite queue. Using standard tests produces
##### - so much error chatter it is hard to unravel what to fix. Standard tests
##### - are also  possibly overly exhaustive which is a bottleneck in turn around.
#####

@testset "Quick AMD" begin
    @info "Testing AMD basics..."

    for arch in archs
        @testset "Quick AMD instantiation [$(typeof(arch))]" begin
            @info "  Quick AMD instantiation [$(typeof(arch))]..."
            N = 3
            topo = (Bounded, Bounded, Bounded)
            FT=Float64
            f(x, y, z) = 1 + exp(x) * sin(y) * tanh(z)
            grid = RectilinearGrid(arch, FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))
            u = XFaceField(grid)
            set!(u, f )
            ans=@allowscalar u .- 0

            # Check against CPU
            grid = RectilinearGrid(CPU(), FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))
            u = XFaceField(grid)
            set!(u, f )
            ans_c=@allowscalar u .- 0

            @test ans ≈ ans_c
        end
    end
end
