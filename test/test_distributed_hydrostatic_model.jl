include("dependencies_for_runtests.jl")

using MPI

# # Distributed model tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# mpiexec -n 4 julia --project test_distributed_models.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
# 
# julia> include("test_distributed_models.jl")
#
# When running the tests this way, uncomment the following line

MPI.Initialized() || MPI.Init()

# to initialize MPI.

using Oceananigans.Operators: hack_cosd
using Oceananigans.DistributedComputations: partition, cpu_architecture, reconstruct_global_grid
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity

Nx = 32
Ny = 32

for arch in archs
    @testset "Testing distributed solid body rotation" begin
        underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 1),
                                                halo = (4, 4, 4),
                                                latitude = (-80, 80),
                                                longitude = (-160, 160),
                                                z = (-1, 0),
                                                radius = 1,
                                                topology=(Bounded, Bounded, Bounded))

        bottom(λ, φ) = -30 < λ < 30 && -40 < φ < 20 ? 0 : - 1

        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
        immersed_active_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        global_underlying_grid = reconstruct_global_grid(underlying_grid)
        global_immersed_grid   = ImmersedBoundaryGrid(global_underlying_grid, GridFittedBottom(bottom))

        for (grid, global_grid) in zip((underlying_grid, immersed_grid, immersed_active_grid), 
                                       (global_underlying_grid, global_immersed_grid, global_immersed_grid))
            
            # We test a couple of different settings
            for (momentum_advection, closure) in zip((WENOVectorInvariant(), VectorInvariant()), 
                                                     (CATKEVerticalDiffusivity(ExplicitTimeDiscretization()), nothing))
                @info "  Testing distributed model: $(typeof(grid).name.wrapper) on $(summary(arch)) with $(summary(closure)) and $(summary(momentum_advection))"
                
                tracers = (:b, :c, :e)

                # "s" for "serial" computation, "p" for "parallel" computation
                ms = rotation_with_shear_test(global_grid; tracers, closure, momentum_advection)
                mp = rotation_with_shear_test(grid;        tracers, closure, momentum_advection)

                us = interior(on_architecture(CPU(), ms.velocities.u))
                vs = interior(on_architecture(CPU(), ms.velocities.v))
                ws = interior(on_architecture(CPU(), ms.velocities.w))
                cs = interior(on_architecture(CPU(), ms.tracers.c))
                ηs = interior(on_architecture(CPU(), ms.free_surface.η))

                cpu_arch = cpu_architecture(arch)

                u = interior(on_architecture(cpu_arch, mp.velocities.u))
                v = interior(on_architecture(cpu_arch, mp.velocities.v))
                w = interior(on_architecture(cpu_arch, mp.velocities.w))
                c = interior(on_architecture(cpu_arch, mp.tracers.c))
                η = interior(on_architecture(cpu_arch, mp.free_surface.η))

                us = partition(us, cpu_arch, size(u))
                vs = partition(vs, cpu_arch, size(v))
                ws = partition(ws, cpu_arch, size(w))
                cs = partition(cs, cpu_arch, size(c))
                ηs = partition(ηs, cpu_arch, size(η))

                atol = eps(eltype(grid))
                rtol = sqrt(eps(eltype(grid)))

                @test all(isapprox(u, us; atol, rtol))
                @test all(isapprox(v, vs; atol, rtol))
                @test all(isapprox(w, ws; atol, rtol))
                @test all(isapprox(c, cs; atol, rtol))
                @test all(isapprox(η, ηs; atol, rtol))
            end
        end
    end
end
