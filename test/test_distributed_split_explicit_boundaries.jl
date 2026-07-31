include("dependencies_for_runtests.jl")

using MPI
MPI.Initialized() || MPI.Init()

using Oceananigans.DistributedComputations: child_architecture, cpu_architecture, partition, ranks, reconstruct_global_grid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, immersed_peripheral_node
using Oceananigans.BoundaryConditions: NormalRadiation, GravityWaveRadiationBoundaryCondition
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: LocalHaloFilling, CompleteHaloFilling

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

# # Distributed open boundaries
#
# `substeps = 8` extends the halo to 7 in Connected directions, which must stay below the local
# extent: this rules out the `Fractional` partitions of `archs` at this grid size.

ηᵢ(x, y, z) = 0.01 * exp(-(x - 0.5)^2 / 0.08) * (1 + 0.2 * cos(2π * y))

function build_open(grid; extend_halos)
    u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(outflow_timescale = 100.0)),
                                    east = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(outflow_timescale = 100.0)))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    west = GravityWaveRadiationBoundaryCondition((0.0, 0.0)),
                                    east = GravityWaveRadiationBoundaryCondition((0.0, 0.0)))

    free_surface = SplitExplicitFreeSurface(grid; substeps=8, extend_halos)

    model = HydrostaticFreeSurfaceModel(grid; free_surface,
                                        boundary_conditions = (u = u_bcs, U = U_bcs),
                                        momentum_advection = nothing,
                                        buoyancy = nothing,
                                        tracers = ())
    set!(model, η = ηᵢ)

    for _ in 1:10
        time_step!(model, 5e-3)
    end

    return model
end

@testset "Distributed open boundary conditions" begin
    open_archs = (Distributed(child_arch; synchronized_communication=false, partition=Partition(4)),
                  Distributed(child_arch; synchronized_communication=false, partition=Partition(2, 2)))

    for arch in open_archs, extend_halos in (true, false)
        cpu_arch = cpu_architecture(arch)

        grid = RectilinearGrid(arch, size=(40, 20, 2), x=(0, 1), y=(0, 1), z=(-1, 0),
                               halo=(4, 4, 2), topology=(Bounded, Periodic, Bounded))
        global_grid = reconstruct_global_grid(grid)

        mp = build_open(grid; extend_halos)         # partitioned
        ms = build_open(global_grid; extend_halos)  # serial reference

        strategy = extend_halos ? LocalHaloFilling : CompleteHaloFilling

        # `LocalHaloFilling` skips the fill that re-radiates the open boundary into a y rank halo
        reproduces_serial = !extend_halos || ranks(arch)[2] == 1

        up = interior(on_architecture(cpu_arch, mp.velocities.u))
        vp = interior(on_architecture(cpu_arch, mp.velocities.v))
        ηp = interior(on_architecture(cpu_arch, mp.free_surface.displacement))

        u_global = interior(on_architecture(CPU(), ms.velocities.u))
        us = partition(u_global, cpu_arch, size(up))
        vs = partition(interior(on_architecture(CPU(), ms.velocities.v)), cpu_arch, size(vp))
        ηs = partition(interior(on_architecture(CPU(), ms.free_surface.displacement)), cpu_arch, size(ηp))

        @testset "open boundaries [extend_halos=$extend_halos, $(typeof(arch.partition))]" begin
            @test mp.free_surface isa SplitExplicitFreeSurface{strategy}
            @test ms.free_surface isa SplitExplicitFreeSurface{strategy}

            if reproduces_serial
                @test all(isapprox.(up, us))
                @test all(isapprox.(vp, vs))
                @test all(isapprox.(ηp, ηs))
            else
                @test_broken all(isapprox.(up, us))
                @test_broken all(isapprox.(vp, vs))
                @test_broken all(isapprox.(ηp, ηs))

                @test maximum(abs, up .- us) < 1e-2 * maximum(abs, us)
                @test maximum(abs, ηp .- ηs) < 1e-2 * maximum(abs, ηs)
            end

            # a solid wall would pin the normal velocity on the global faces to zero
            @test maximum(abs, up) > 0
            @test maximum(abs, u_global[1, :, :])   > 0
            @test maximum(abs, u_global[end, :, :]) > 0
        end
    end
end
