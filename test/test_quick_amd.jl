include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")
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
        @testset "Quick AMD set field [$(typeof(arch))]" begin

            @info "  Quick AMD set field from func [$(typeof(arch))]..."
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

            @info "  Quick AMD set field from GPU array [$(typeof(arch))]..."
            N = 3
            topo = (Bounded, Bounded, Bounded)
            FT=Float64
            grid = RectilinearGrid(arch, FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))
            Nx, Ny, Nz = size(grid)
            f = rand(Nx, Ny, Nz)
            f_d = ROCArray(f)
            default_bcs = FieldBoundaryConditions()
            u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
            u = CenterField(grid,boundary_conditions=u_bcs)
            set!(u, f_d )
            ans=@allowscalar u .- 0
            # Check against CPU
            grid = RectilinearGrid(CPU(), FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))
            Nx, Ny, Nz = size(grid)
            u = CenterField(grid,boundary_conditions=u_bcs)
            set!(u, f )
            ans_c=@allowscalar u .- 0
            @test ans ≈ ans_c

            @info "  Quick AMD set field from CPU array [$(typeof(arch))]..."
            N = 3
            topo = (Bounded, Bounded, Bounded)
            FT=Float64
            grid = RectilinearGrid(arch, FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))
            Nx, Ny, Nz = size(grid)
            f = rand(Nx, Ny, Nz)
            default_bcs = FieldBoundaryConditions()
            u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
            u = CenterField(grid,boundary_conditions=u_bcs)
            set!(u, f )
            ans=@allowscalar u .- 0
            # Check against CPU
            grid = RectilinearGrid(CPU(), FT, topology=topo, size=(N, N, N), x=(-1, 1), y=(0, 2π), z=(-1, 1))
            default_bcs = FieldBoundaryConditions()
            u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
            u = CenterField(grid,boundary_conditions=u_bcs)
            set!(u, f )
            ans_c=@allowscalar u .- 0

            @test ans ≈ ans_c

        end
    end
end
