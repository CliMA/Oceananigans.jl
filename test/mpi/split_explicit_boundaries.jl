include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:multiple)

using Oceananigans.DistributedComputations: child_architecture, cpu_architecture, partition, reconstruct_global_grid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, immersed_peripheral_node
using Oceananigans.BoundaryConditions: NormalRadiation, ObliqueRadiation, GravityWaveRadiationBoundaryCondition
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: LocalHaloFilling, CompleteHaloFilling

# # Distributed SplitExplicitFreeSurface boundary handling
#
# Run on 4 ranks, e.g.
#
#   $ MPI_TEST=true mpiexec -n 4 julia --project test/mpi/split_explicit_boundaries.jl
#
# The point is that the barotropic corrector mask must behave identically across rank
# boundaries: a partitioned domain only differs from the serial one in its *Connected*
# (inter-rank) halos, which are active — so the mask must NOT fire there — while the global
# domain edges (closed walls / open boundaries) must be handled exactly as in serial. Any
# regression to `inactive_node` (which is true in the exterior halo) would diverge here.

bump(x, y) = -0.5 - 0.4 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.05)

bᵢ(x, y, z) = z + 0.02 * sin(2π * x) * cos(2π * y)

function build(grid; substeps=10, Δt=2e-3)
    free_surface = SplitExplicitFreeSurface(grid; substeps)
    model = HydrostaticFreeSurfaceModel(grid; free_surface, buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=bᵢ)
    for _ in 1:10
        time_step!(model, Δt)
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

# # More substeps than local cells
#
# `substeps = 30` extends the halo beyond the local size of every partition in `archs` at this grid size,
# so the halo is capped at the smallest local size and refilled during the substepping. `Δt = 0.05` gives a
# barotropic Courant number large enough for a stale halo to reach the interior.

@testset "Distributed SplitExplicitFreeSurface with more substeps than local cells" begin
    for arch in archs
        cpu_arch = cpu_architecture(arch)

        grid = RectilinearGrid(arch, size=(32, 32, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                               halo=(3, 3, 3), topology=(Bounded, Bounded, Bounded))

        mp = build(grid; substeps=30, Δt=0.05)
        ms = build(reconstruct_global_grid(grid); substeps=30, Δt=0.05)

        up = interior(on_architecture(cpu_arch, mp.velocities.u))
        vp = interior(on_architecture(cpu_arch, mp.velocities.v))
        ηp = interior(on_architecture(cpu_arch, mp.free_surface.displacement))

        us = partition(interior(on_architecture(CPU(), ms.velocities.u)), cpu_arch, size(up))
        vs = partition(interior(on_architecture(CPU(), ms.velocities.v)), cpu_arch, size(vp))
        ηs = partition(interior(on_architecture(CPU(), ms.free_surface.displacement)), cpu_arch, size(ηp))

        @testset "serial vs distributed [$(typeof(arch.partition))]" begin
            @test all(isapprox.(up, us))
            @test all(isapprox.(vp, vs))
            @test all(isapprox.(ηp, ηs))
        end
    end
end

# # Distributed open boundaries
#
# `substeps = 30` extends the halo beyond the local size of both partitions, so the halo is capped and refilled
# during the substepping, with a `Δt` large enough for a stale halo to reach the interior.

ηᵢ(x, y, z) = 0.01 * exp(-(x - 0.5)^2 / 0.08) * (1 + 0.2 * cos(2π * y))

function build_open(grid; extend_halos, substeps, Δt)
    u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(outflow_timescale = 100.0)),
                                    east = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(outflow_timescale = 100.0)))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    west = GravityWaveRadiationBoundaryCondition((0.0, 0.0)),
                                    east = GravityWaveRadiationBoundaryCondition((0.0, 0.0)))

    free_surface = SplitExplicitFreeSurface(grid; substeps, extend_halos)

    model = HydrostaticFreeSurfaceModel(grid; free_surface,
                                        boundary_conditions = (u = u_bcs, U = U_bcs),
                                        momentum_advection = nothing,
                                        buoyancy = nothing,
                                        tracers = ())
    set!(model, η = ηᵢ)

    for _ in 1:10
        time_step!(model, Δt)
    end

    return model
end

@testset "Distributed open boundary conditions" begin
    open_archs = (Distributed(child_arch; synchronized_communication=false, partition=Partition(4)),
                  Distributed(child_arch; synchronized_communication=false, partition=Partition(2, 2)))

    for arch in open_archs, extend_halos in (true, false), (substeps, Δt) in ((8, 5e-3), (30, 5e-2))
        cpu_arch = cpu_architecture(arch)

        grid = RectilinearGrid(arch, size=(40, 20, 2), x=(0, 1), y=(0, 1), z=(-1, 0),
                               halo=(4, 4, 2), topology=(Open, Periodic, Bounded))
        global_grid = reconstruct_global_grid(grid)

        mp = build_open(grid; extend_halos, substeps, Δt)         # partitioned
        ms = build_open(global_grid; extend_halos, substeps, Δt)  # serial reference

        strategy = extend_halos ? LocalHaloFilling : CompleteHaloFilling

        up = interior(on_architecture(cpu_arch, mp.velocities.u))
        vp = interior(on_architecture(cpu_arch, mp.velocities.v))
        ηp = interior(on_architecture(cpu_arch, mp.free_surface.displacement))

        u_global = interior(on_architecture(CPU(), ms.velocities.u))
        us = partition(u_global, cpu_arch, size(up))
        vs = partition(interior(on_architecture(CPU(), ms.velocities.v)), cpu_arch, size(vp))
        ηs = partition(interior(on_architecture(CPU(), ms.free_surface.displacement)), cpu_arch, size(ηp))

        @testset "open boundaries [extend_halos=$extend_halos, substeps=$substeps, $(typeof(arch.partition))]" begin
            @test mp.free_surface isa SplitExplicitFreeSurface{strategy}
            @test ms.free_surface isa SplitExplicitFreeSurface{strategy}

            @test all(isapprox.(up, us))
            @test all(isapprox.(vp, vs))
            @test all(isapprox.(ηp, ηs))

            # a solid wall would pin the normal velocity on the global faces to zero
            @test maximum(abs, up) > 0
            @test maximum(abs, u_global[1, :, :])   > 0
            @test maximum(abs, u_global[end, :, :]) > 0
        end
    end
end

# # Boundary faces on the edge ranks of a distributed Bounded dimension
#
# On the left-edge rank the topology is `RightConnected`, but face 1 is still the domain boundary and must
# not be stepped with the interior. Rotation of a sheared `v` gives the boundary face a depth-varying tendency
# that the barotropic correction does not remove, and the anchored radiation fill reads the boundary value.

vᵢ(x, y, z) = 0.01 * (1 + 2z)

function build_rotating_open(grid; timestepper)
    u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(outflow_timescale = 100.0)),
                                    east = NormalFlowBoundaryCondition(0; scheme = NormalRadiation(outflow_timescale = 100.0)))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    west = GravityWaveRadiationBoundaryCondition((0.0, 0.0)),
                                    east = GravityWaveRadiationBoundaryCondition((0.0, 0.0)))

    free_surface = SplitExplicitFreeSurface(grid; substeps=8, extend_halos=false)

    model = HydrostaticFreeSurfaceModel(grid; free_surface, timestepper,
                                        coriolis = FPlane(f = 10),
                                        boundary_conditions = (u = u_bcs, U = U_bcs),
                                        momentum_advection = nothing,
                                        buoyancy = nothing,
                                        tracers = ())
    set!(model, η = ηᵢ, v = vᵢ)

    for _ in 1:10
        time_step!(model, 5e-3)
    end

    return model
end

@testset "Distributed open boundary faces are not stepped" begin
    open_archs = (Distributed(child_arch; synchronized_communication=false, partition=Partition(4)),
                  Distributed(child_arch; synchronized_communication=false, partition=Partition(2, 2)))

    for arch in open_archs, timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        cpu_arch = cpu_architecture(arch)

        grid = RectilinearGrid(arch, size=(40, 20, 2), x=(0, 1), y=(0, 1), z=(-1, 0),
                               halo=(4, 4, 2), topology=(Bounded, Periodic, Bounded))
        global_grid = reconstruct_global_grid(grid)

        mp = build_rotating_open(grid; timestepper)
        ms = build_rotating_open(global_grid; timestepper)

        up = interior(on_architecture(cpu_arch, mp.velocities.u))
        vp = interior(on_architecture(cpu_arch, mp.velocities.v))
        ηp = interior(on_architecture(cpu_arch, mp.free_surface.displacement))

        us = partition(interior(on_architecture(CPU(), ms.velocities.u)), cpu_arch, size(up))
        vs = partition(interior(on_architecture(CPU(), ms.velocities.v)), cpu_arch, size(vp))
        ηs = partition(interior(on_architecture(CPU(), ms.free_surface.displacement)), cpu_arch, size(ηp))

        @testset "rotating open boundaries [$timestepper, $(typeof(arch.partition))]" begin
            @test all(isapprox.(up, us))
            @test all(isapprox.(vp, vs))
            @test all(isapprox.(ηp, ηs))
        end
    end
end

# # Distributed oblique radiation
#
# A tracer leaving through south and north boundaries with `ObliqueRadiation`, varying along them; `x` is
# periodic, so the tangential differences cross the rank edges and the periodic ends.

function build_oblique(grid)
    scheme = ObliqueRadiation(inflow_timescale = 100.0, outflow_timescale = 100.0)

    v_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0; scheme),
                                    north = NormalFlowBoundaryCondition(0; scheme))

    V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
                                    south = GravityWaveRadiationBoundaryCondition((0.0, 0.0)),
                                    north = GravityWaveRadiationBoundaryCondition((0.0, 0.0)))

    c_bcs = FieldBoundaryConditions(south = ValueBoundaryCondition(0; scheme),
                                    north = ValueBoundaryCondition(0; scheme))

    free_surface = SplitExplicitFreeSurface(grid; substeps=8)

    model = HydrostaticFreeSurfaceModel(grid; free_surface,
                                        boundary_conditions = (v = v_bcs, V = V_bcs, c = c_bcs),
                                        momentum_advection = nothing,
                                        buoyancy = nothing,
                                        tracers = :c)

    set!(model, η = (x, y, z) -> 0.01 * exp(-(y - 0.5)^2 / 0.08) * (1 + 0.2 * cos(2π * x)),
                c = (x, y, z) -> (1 + 0.5 * sin(2π * x)) * y)

    for _ in 1:60
        time_step!(model, 5e-3)
    end

    return model
end

@testset "Distributed oblique radiation" begin
    oblique_archs = (Distributed(child_arch; synchronized_communication=false, partition=Partition(4)),
                     Distributed(child_arch; synchronized_communication=false, partition=Partition(2, 2)))

    for arch in oblique_archs
        cpu_arch = cpu_architecture(arch)

        grid = RectilinearGrid(arch, size=(40, 20, 2), x=(0, 1), y=(0, 1), z=(-1, 0),
                               halo=(4, 4, 2), topology=(Periodic, Open, Bounded))

        mp = build_oblique(grid)                           # partitioned
        ms = build_oblique(reconstruct_global_grid(grid))  # serial reference

        cp = interior(on_architecture(cpu_arch, mp.tracers.c))
        cs = partition(interior(on_architecture(CPU(), ms.tracers.c)), cpu_arch, size(cp))

        @testset "oblique radiation [$(typeof(arch.partition))]" begin
            @test all(isapprox.(cp, cs))
        end
    end
end

# # Distributed tangential velocity
#
# The velocity along the open boundaries radiates with the phase speed of the velocity across them, and is nudged toward
# an exterior value that varies along them and with depth, so that the neighbours across the rank edges and the periodic
# ends matter. With a split-explicit free surface the barotropic correction replaces the depth mean of the velocity at the
# boundary, so the exterior value has a vertical shear.

function build_oblique_tangential(grid; split_explicit)
    scheme = ObliqueRadiation(inflow_timescale = 100.0, outflow_timescale = 100.0)
    nudged = ObliqueRadiation(inflow_timescale = 0.05, outflow_timescale = 0.05)
    exterior(x, z, t) = 0.3 * (1 + sin(2π * 3 * x)) * (1 + 0.5 * (z + 0.5))

    u_bcs = FieldBoundaryConditions(south = ValueBoundaryCondition(exterior; scheme = nudged),
                                    north = ValueBoundaryCondition(exterior; scheme = nudged))

    v_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(0; scheme),
                                    north = NormalFlowBoundaryCondition(0; scheme))

    boundary_conditions = (u = u_bcs, v = v_bcs)
    free_surface = nothing

    if split_explicit
        V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
                                        south = GravityWaveRadiationBoundaryCondition((0.0, 0.0)),
                                        north = GravityWaveRadiationBoundaryCondition((0.0, 0.0)))
        boundary_conditions = merge(boundary_conditions, (V = V_bcs,))
        free_surface = SplitExplicitFreeSurface(grid; substeps = 8)
    end

    model = HydrostaticFreeSurfaceModel(grid; free_surface, boundary_conditions,
                                        momentum_advection = Centered(),
                                        buoyancy = nothing,
                                        tracers = ())

    set!(model, u = (x, y, z) -> 0.1 * (1 + 0.5 * sin(2π * x)) * (1 + y),
                v = (x, y, z) -> 0.1 * y * (1 + 0.8 * sin(2π * 13 * x) + 0.5 * cos(2π * 7 * x)))

    for _ in 1:60
        time_step!(model, 5e-3)
    end

    return model
end

@testset "Distributed oblique radiation of the tangential velocity" begin
    for arch in (Distributed(child_arch; synchronized_communication=false, partition=Partition(4)),
                 Distributed(child_arch; synchronized_communication=false, partition=Partition(2, 2))),
        split_explicit in (false, true)

        cpu_arch = cpu_architecture(arch)

        grid = RectilinearGrid(arch, size=(40, 20, 2), x=(0, 1), y=(0, 1), z=(-1, 0),
                               halo=(4, 4, 2), topology=(Periodic, Open, Bounded))

        mp = build_oblique_tangential(grid; split_explicit)                           # partitioned
        ms = build_oblique_tangential(reconstruct_global_grid(grid); split_explicit)  # serial reference

        up = interior(on_architecture(cpu_arch, mp.velocities.u))
        vp = interior(on_architecture(cpu_arch, mp.velocities.v))
        us = partition(interior(on_architecture(CPU(), ms.velocities.u)), cpu_arch, size(up))
        vs = partition(interior(on_architecture(CPU(), ms.velocities.v)), cpu_arch, size(vp))

        @testset "tangential oblique radiation [$(typeof(arch.partition)), split_explicit=$split_explicit]" begin
            @test all(isapprox.(up, us; rtol = 1e-12))
            @test all(isapprox.(vp, vs; rtol = 1e-12))
        end
    end
end
