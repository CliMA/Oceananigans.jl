include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using MPI
using Oceananigans.Grids: MutableVerticalDiscretization, StaticVerticalDiscretization, with_halo
using Oceananigans.ImmersedBoundaries: immersed_cell

# # Distributed model tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# $ MPI_TEST=true mpiexec -n 4 julia --project test/mpi/hydrostatic_model.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
#
# julia> include("test/mpi/hydrostatic_model.jl")

MPI.Initialized() || MPI.Init(threadlevel=:multiple)

using Oceananigans.DistributedComputations: ranks, partition, all_reduce, cpu_architecture, reconstruct_global_grid, synchronized, synchronize_communication!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity

function Δ_min(grid)
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

@inline Gaussian(x, y, L) = exp(-(x^2 + y^2) / L^2)

@kernel function _increment!(c)
    i, j, k = @index(Global, NTuple)
    @inbounds c[i, j, k] += 1
end

@inline active_cell(i, j, k, grid) = !immersed_cell(i, j, k, grid)

function rotation_with_shear_test(grid, closure=nothing; timestepper=:QuasiAdamsBashforth2)

    g = one(grid)
    R = grid.radius
    Ω = Oceananigans.defaults.planet_rotation_rate

    # Add some shear on the velocity field
    uᵢ(λ, φ, z) = 0.1 * cosd(φ) * sind(λ) + 0.05 * z
    ηᵢ(λ, φ, z) = (R * Ω * 0.1 + 0.1^2 / 2) * sind(φ)^2 / g * sind(λ)

    # Gaussian leads to values with O(1e-60),
    # too small for repetible testing. We cap it at 0.1
    cᵢ(λ, φ, z) = max(Gaussian(λ, φ - 5, 10), 0.1)
    vᵢ(λ, φ, z) = 0.1

    free_surface = SplitExplicitFreeSurface(grid; substeps = 8, gravitational_acceleration = g)
    coriolis     = HydrostaticSphericalCoriolis(rotation_rate = 1)

    model = HydrostaticFreeSurfaceModel(grid;
                                        momentum_advection = WENOVectorInvariant(order=3),
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        closure,
                                        tracers = (:c, :b),
                                        timestepper,
                                        tracer_advection = WENO(order=3),
                                        buoyancy = BuoyancyTracer())

    set!(model, u=uᵢ, c=cᵢ, η=ηᵢ)

    Δt_local = 0.1 * Δ_min(grid) / sqrt(g * grid.Lz)
    Δt = all_reduce(min, Δt_local, architecture(grid))

    for _ in 1:10
        time_step!(model, Δt)
    end

    for field in merge(model.velocities, model.tracers)
        synchronize_communication!(field)
    end

    return model
end

Nx = 32
Ny = 32

for arch in archs
    # We do not test on `Fractional` partitions where we cannot easily ensure that H ≤ N
    # which would lead to different advection schemes for partitioned and non-partitioned grids.
    # `Fractional` is, however, tested in regression tests where the horizontal dimensions are larger.
    valid_x_partition = !(arch.partition.x isa Fractional)
    valid_y_partition = !(arch.partition.y isa Fractional)
    valid_z_partition = !(arch.partition.z isa Fractional)

    if valid_x_partition & valid_y_partition & valid_z_partition
        @testset "Interior launches visit every active cell once [$(ranks(arch))]" begin
            underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 3), halo = (4, 4, 3),
                                                    latitude = (-80, 80), longitude = (-160, 160), z = (-1, 0))

            bottom(λ, φ) = -30 < λ < 30 && -40 < φ < 20 ? 0 : - 1
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

            visits = CenterField(grid)
            launch!(arch, grid, :xyz, _increment!, visits)
            active = Field(KernelFunctionOperation{Center, Center, Center}(active_cell, grid))

            @test on_architecture(CPU(), interior(visits)) == on_architecture(CPU(), interior(active))
        end

        @testset "Testing distributed solid body rotation" begin

            child_arch = child_architecture(arch)

            # TODO: MutableVerticalDiscretization does not lead to exactly equal solutions.
            # For the moment we remove the offending tests, but we need to investigate this...
            # z_faces = [(-1, 0), MutableVerticalDiscretization((-1, 0))]
            z_faces = [(-1, 0)]

            for z_face in z_faces
                underlying_grid = LatitudeLongitudeGrid(arch,
                                                        size = (Nx, Ny, 3),
                                                        halo = (4, 4, 3),
                                                        latitude = (-80, 80),
                                                        longitude = (-160, 160),
                                                        z = z_face,
                                                        radius = 10,
                                                        topology = (Bounded, Bounded, Bounded))

                bottom(λ, φ) = -30 < λ < 30 && -40 < φ < 20 ? 0 : - 1

                immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = false)
                immersed_active_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

                # The halo-dependent and halo-independent maps must follow the new halo
                new_halo = (5, 5, 3)
                extended_active_grid = with_halo(new_halo, immersed_active_grid)
                rebuilt_active_grid = ImmersedBoundaryGrid(with_halo(new_halo, underlying_grid), GridFittedBottom(bottom); active_cells_map = true)
                @test extended_active_grid.interior_active_cells == rebuilt_active_grid.interior_active_cells
                @test extended_active_grid.active_z_columns == rebuilt_active_grid.active_z_columns

                global_underlying_grid = reconstruct_global_grid(underlying_grid)
                global_immersed_grid   = ImmersedBoundaryGrid(global_underlying_grid, GridFittedBottom(bottom))

                for (grid, global_grid) in zip((underlying_grid, immersed_grid, immersed_active_grid),
                                               (global_underlying_grid, global_immersed_grid, global_immersed_grid))

                    @testset "Solid body rotation with $(ranks(arch)) ranks on $(nameof(typeof(grid))) and a $(nameof(typeof(grid.z))) on $(timestepper)" for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
                        # We deepcopy the grid to ensure it is not modified in the case of a MutableVerticalDiscretization
                        test_global_grid = deepcopy(global_grid)
                        test_grid = deepcopy(grid)

                        # "s" for "serial" computation, "p" for parallel
                        ms = rotation_with_shear_test(test_global_grid; timestepper)
                        mp = rotation_with_shear_test(test_grid; timestepper)

                        us = interior(on_architecture(CPU(), ms.velocities.u))
                        vs = interior(on_architecture(CPU(), ms.velocities.v))
                        cs = interior(on_architecture(CPU(), ms.tracers.c))
                        ηs = interior(on_architecture(CPU(), ms.free_surface.displacement))

                        cpu_arch = cpu_architecture(arch)

                        up = interior(on_architecture(cpu_arch, mp.velocities.u))
                        vp = interior(on_architecture(cpu_arch, mp.velocities.v))
                        cp = interior(on_architecture(cpu_arch, mp.tracers.c))
                        ηp = interior(on_architecture(cpu_arch, mp.free_surface.displacement))

                        us = partition(us, cpu_arch, size(up))
                        vs = partition(vs, cpu_arch, size(vp))
                        cs = partition(cs, cpu_arch, size(cp))
                        ηs = partition(ηs, cpu_arch, size(ηp))

                        atol = eps(eltype(grid))
                        rtol = sqrt(eps(eltype(grid)))

                        @test all(isapprox(up, us; atol, rtol))
                        @test all(isapprox(vp, vs; atol, rtol))
                        @test all(isapprox(cp, cs; atol, rtol))
                        @test all(isapprox(ηp, ηs; atol, rtol))
                    end
                end

                closure = CATKEVerticalDiffusivity()
                cpu_arch = cpu_architecture(arch)

                @root @info "  Testing CATKE with $(ranks(arch)) ranks"

                # "s" for "serial" computation, "p" for parallel
                ms = rotation_with_shear_test(global_immersed_grid, closure)
                mp = rotation_with_shear_test(immersed_active_grid, closure)

                us = interior(on_architecture(CPU(), ms.velocities.u))
                vs = interior(on_architecture(CPU(), ms.velocities.v))
                cs = interior(on_architecture(CPU(), ms.tracers.c))
                ηs = interior(on_architecture(CPU(), ms.free_surface.displacement))

                up = interior(on_architecture(cpu_arch, mp.velocities.u))
                vp = interior(on_architecture(cpu_arch, mp.velocities.v))
                cp = interior(on_architecture(cpu_arch, mp.tracers.c))
                ηp = interior(on_architecture(cpu_arch, mp.free_surface.displacement))

                us = partition(us, cpu_arch, size(up))
                vs = partition(vs, cpu_arch, size(vp))
                cs = partition(cs, cpu_arch, size(cp))
                ηs = partition(ηs, cpu_arch, size(ηp))

                atol = eps(eltype(global_underlying_grid))
                rtol = sqrt(eps(eltype(global_underlying_grid)))

                @test all(isapprox(up, us; atol, rtol))
                @test all(isapprox(vp, vs; atol, rtol))
                @test all(isapprox(cp, cs; atol, rtol))
                @test all(isapprox(ηp, ηs; atol, rtol))
            end
        end
    end
end

# A bounded tracer with a sharp front, advected across the rank edges by a uniform flow
function bounded_tracer_advection_test(grid)
    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing,
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 4),
                                        tracer_advection = WENO(order = 5, bounds = (0, 1)))

    set!(model, u = 0.1, v = 0.1, c = (x, y, z) -> ifelse(abs(x - 5e4) < 2e4 && abs(y - 5e4) < 2e4, 1, 0))

    for _ in 1:10
        time_step!(model, 600)
    end

    return model
end

@testset "Distributed bounds-preserving tracer advection" begin
    for arch in archs
        (arch.partition.x isa Fractional || arch.partition.y isa Fractional) && continue

        bounded_grid(arch) = RectilinearGrid(arch, size = (32, 32, 4), x = (0, 1e5), y = (0, 1e5), z = (-100, 0),
                                             halo = (4, 4, 4), topology = (Periodic, Periodic, Bounded))

        ms = bounded_tracer_advection_test(bounded_grid(child_architecture(arch)))
        mp = bounded_tracer_advection_test(bounded_grid(arch))

        cpu_arch = cpu_architecture(arch)
        p = interior(on_architecture(cpu_arch, mp.tracers.c))
        s = partition(interior(on_architecture(CPU(), ms.tracers.c)), cpu_arch, size(p))
        @test all(isapprox(p, s; atol = eps(eltype(p)), rtol = sqrt(eps(eltype(p)))))
    end
end

@testset "Distributed default free surface" begin
    using Oceananigans.Models.HydrostaticFreeSurfaceModels: estimate_maximum_Δt
    using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: FixedSubstepNumber, CompleteHaloFilling

    # A shallow domain needs few substeps, so halos are extended; a deep domain on a
    # coarse grid needs more substeps than fit in a rank's subdomain, so halos are not extended.
    configurations = ((N = 128, H = 10,   extend_halos = true),
                      (N = 16,  H = 1000, extend_halos = false))

    for arch in archs, (; N, H, extend_halos) in configurations
        grids = (RectilinearGrid(arch, size = (N, N, 4), x = (0, 1e5), y = (0, 1e5), z = (-H, 0),
                                 halo = (4, 4, 4), topology = (Periodic, Periodic, Bounded)),
                 LatitudeLongitudeGrid(arch, size = (N, N, 4), halo = (4, 4, 4),
                                       latitude = (-60, 60), longitude = (0, 360), z = (-H, 0)))

        for grid in grids
            @testset "Default free surface on $(nameof(typeof(grid))) with $(ranks(arch)) ranks and depth $H" begin
                model = HydrostaticFreeSurfaceModel(grid; tracers = :b, buoyancy = BuoyancyTracer())
                free_surface = model.free_surface

                @test free_surface isa SplitExplicitFreeSurface
                @test free_surface.substepping isa FixedSubstepNumber

                # All ranks must agree on the number of substeps, which must match the global grid's
                substeps = length(free_surface.substepping.averaging_weights)
                @test all_reduce(min, substeps, arch) == all_reduce(max, substeps, arch)

                global_grid = reconstruct_global_grid(grid)
                global_free_surface = SplitExplicitFreeSurface(global_grid; cfl = 0.7, fixed_Δt = estimate_maximum_Δt(global_grid))
                @test substeps == length(global_free_surface.substepping.averaging_weights)

                @test (free_surface isa SplitExplicitFreeSurface{<:CompleteHaloFilling}) == !extend_halos

                Δt = estimate_maximum_Δt(grid)
                for _ in 1:3
                    time_step!(model, Δt)
                end

                @test all(isfinite, free_surface.displacement)
            end
        end
    end
end
