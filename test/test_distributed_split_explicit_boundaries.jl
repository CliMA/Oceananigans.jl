include("dependencies_for_runtests.jl")

using MPI
MPI.Initialized() || MPI.Init()

using Oceananigans.DistributedComputations: child_architecture, cpu_architecture, partition, reconstruct_global_grid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, immersed_peripheral_node

# # Distributed SplitExplicitFreeSurface boundary handling
#
# Run on 4 ranks, e.g.
#
#   $ MPI_TEST=true mpiexec -n 4 julia --project test_distributed_split_explicit_boundaries.jl
#
# The point is that the barotropic corrector mask must behave identically across rank
# boundaries: a partitioned domain only differs from the serial one in its *Connected*
# (inter-rank) halos, which are active — so the mask must NOT fire there — while the global
# domain edges (closed walls / open boundaries) must be handled exactly as in serial. Any
# regression to `inactive_node` (which is true in the exterior halo) would diverge here.

bump(x, y) = -0.5 - 0.4 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.05)

bᵢ(x, y, z) = z + 0.02 * sin(2π * x) * cos(2π * y)

function build(grid)
    free_surface = SplitExplicitFreeSurface(grid; substeps=10)
    model = HydrostaticFreeSurfaceModel(grid; free_surface, buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=bᵢ)
    for _ in 1:10
        time_step!(model, 2e-3)
    end
    return model
end

@testset "Distributed SplitExplicitFreeSurface boundaries" begin
    for arch in archs
        child_arch = child_architecture(arch)
        cpu_arch   = cpu_architecture(arch)

        for immersed in (false, true)
            underlying = RectilinearGrid(arch, size=(64, 64, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                                         halo=(4, 4, 4), topology=(Bounded, Bounded, Bounded))
            grid = immersed ? ImmersedBoundaryGrid(underlying, GridFittedBottom(bump)) : underlying

            global_underlying = reconstruct_global_grid(underlying)
            global_grid = immersed ? ImmersedBoundaryGrid(global_underlying, GridFittedBottom(bump)) : global_underlying

            mp = build(grid)         # partitioned (distributed)
            ms = build(global_grid)  # serial reference

            up = interior(on_architecture(cpu_arch, mp.velocities.u))
            vp = interior(on_architecture(cpu_arch, mp.velocities.v))
            ηp = interior(on_architecture(cpu_arch, mp.free_surface.displacement))

            us = partition(interior(on_architecture(CPU(), ms.velocities.u)), cpu_arch, size(up))
            vs = partition(interior(on_architecture(CPU(), ms.velocities.v)), cpu_arch, size(vp))
            ηs = partition(interior(on_architecture(CPU(), ms.free_surface.displacement)), cpu_arch, size(ηp))

            @testset "serial vs distributed [immersed=$immersed, $(typeof(arch.partition))]" begin
                @test all(isapprox.(up, us))
                @test all(isapprox.(vp, vs))
                @test all(isapprox.(ηp, ηs))
            end

            # Immersed velocities are zeroed in the distributed model too.
            if immersed
                u = mp.velocities.u
                f, c = Face(), Center()
                Nx, Ny, Nz = size(grid)
                @testset "distributed immersed velocities zero [immersed=$immersed]" begin
                    @test all(u[i, j, k] == 0 for i in 1:Nx+1, j in 1:Ny, k in 1:Nz
                              if immersed_peripheral_node(i, j, k, grid, f, c, c))
                end
            end
        end
    end
end
