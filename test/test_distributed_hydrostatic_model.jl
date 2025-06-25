include("dependencies_for_runtests.jl")

using MPI

# # Distributed model tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# $ MPI_TEST=true mpiexec -n 4 julia --project test_distributed_hydrostatic_model.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
#
# julia> include("test_distributed_hydrostatic_model.jl")

MPI.Initialized() || MPI.Init()

using Oceananigans.Operators: hack_cosd
using Oceananigans.DistributedComputations: ranks, partition, all_reduce, cpu_architecture, reconstruct_global_grid, synchronized
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity

function Δ_min(grid)
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

@inline Gaussian(x, y, L) = exp(-(x^2 + y^2) / L^2)

function rotation_with_shear_test(grid, closure=nothing)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 8, gravitational_acceleration = 1)
    coriolis     = HydrostaticSphericalCoriolis(rotation_rate = 1)

    tracers = if closure isa CATKEVerticalDiffusivity
        (:c, :b, :e)
    else
        (:c, :b)
    end

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = WENOVectorInvariant(order=3),
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        closure,
                                        tracers,
                                        tracer_advection = WENO(order=3),
                                        buoyancy = BuoyancyTracer())

    g = model.free_surface.gravitational_acceleration
    R = grid.radius
    Ω = model.coriolis.rotation_rate

    # Add some shear on the velocity field
    uᵢ(λ, φ, z) = 0.1 * cosd(φ) * sind(λ) + 0.05 * z
    ηᵢ(λ, φ, z) = (R * Ω * 0.1 + 0.1^2 / 2) * sind(φ)^2 / g * sind(λ)

    # Gaussian leads to values with O(1e-60),
    # too small for repetible testing. We cap it at 0.1
    cᵢ(λ, φ, z) = max(Gaussian(λ, φ - 5, 10), 0.1)
    vᵢ(λ, φ, z) = 0.1

    set!(model, u=uᵢ, η=ηᵢ, c=cᵢ)

    Δt_local = 0.1 * Δ_min(grid) / sqrt(g * grid.Lz)
    Δt = all_reduce(min, Δt_local, architecture(grid))

    for _ in 1:10
        time_step!(model, Δt)
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
        @testset "Testing distributed solid body rotation" begin
            underlying_grid = LatitudeLongitudeGrid(arch,
                                                    size = (Nx, Ny, 3),
                                                    halo = (4, 4, 3),
                                                    latitude = (-80, 80),
                                                    longitude = (-160, 160),
                                                    z = (-1, 0),
                                                    radius = 1,
                                                    topology = (Bounded, Bounded, Bounded))

            bottom(λ, φ) = -30 < λ < 30 && -40 < φ < 20 ? 0 : - 1

            immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = false)
            immersed_active_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

            global_underlying_grid = reconstruct_global_grid(underlying_grid)
            global_immersed_grid   = ImmersedBoundaryGrid(global_underlying_grid, GridFittedBottom(bottom))

            for (grid, global_grid) in zip((underlying_grid, immersed_grid, immersed_active_grid),
                                           (global_underlying_grid, global_immersed_grid, global_immersed_grid))
                if arch.local_rank == 0
                    @info "  Testing distributed solid body rotation with $(ranks(arch)) ranks on $(typeof(grid).name.wrapper)"
                end

                # "s" for "serial" computation, "p" for parallel
                ms = rotation_with_shear_test(global_grid)
                mp = rotation_with_shear_test(grid)

                us = interior(on_architecture(CPU(), ms.velocities.u))
                vs = interior(on_architecture(CPU(), ms.velocities.v))
                ws = interior(on_architecture(CPU(), ms.velocities.w))
                cs = interior(on_architecture(CPU(), ms.tracers.c))
                ηs = interior(on_architecture(CPU(), ms.free_surface.η))

                cpu_arch = cpu_architecture(arch)

                up = interior(on_architecture(cpu_arch, mp.velocities.u))
                vp = interior(on_architecture(cpu_arch, mp.velocities.v))
                wp = interior(on_architecture(cpu_arch, mp.velocities.w))
                cp = interior(on_architecture(cpu_arch, mp.tracers.c))
                ηp = interior(on_architecture(cpu_arch, mp.free_surface.η))

                us = partition(us, cpu_arch, size(up))
                vs = partition(vs, cpu_arch, size(vp))
                ws = partition(ws, cpu_arch, size(wp))
                cs = partition(cs, cpu_arch, size(cp))
                ηs = partition(ηs, cpu_arch, size(ηp))

                atol = eps(eltype(grid))
                rtol = sqrt(eps(eltype(grid)))

                @test all(isapprox(up, us; atol, rtol))
                @test all(isapprox(vp, vs; atol, rtol))
                @test all(isapprox(wp, ws; atol, rtol))
                @test all(isapprox(cp, cs; atol, rtol))
                @test all(isapprox(ηp, ηs; atol, rtol))
            end

            # CATKE works only with synchronized communication at the moment
            arch    = synchronized(arch)
            closure = CATKEVerticalDiffusivity()

            # "s" for "serial" computation, "p" for parallel
            ms = rotation_with_shear_test(global_underlying_grid, closure)
            mp = rotation_with_shear_test(underlying_grid, closure)

            us = interior(on_architecture(CPU(), ms.velocities.u))
            vs = interior(on_architecture(CPU(), ms.velocities.v))
            ws = interior(on_architecture(CPU(), ms.velocities.w))
            cs = interior(on_architecture(CPU(), ms.tracers.c))
            ηs = interior(on_architecture(CPU(), ms.free_surface.η))

            cpu_arch = cpu_architecture(arch)

            up = interior(on_architecture(cpu_arch, mp.velocities.u))
            vp = interior(on_architecture(cpu_arch, mp.velocities.v))
            wp = interior(on_architecture(cpu_arch, mp.velocities.w))
            cp = interior(on_architecture(cpu_arch, mp.tracers.c))
            ηp = interior(on_architecture(cpu_arch, mp.free_surface.η))

            us = partition(us, cpu_arch, size(up))
            vs = partition(vs, cpu_arch, size(vp))
            ws = partition(ws, cpu_arch, size(wp))
            cs = partition(cs, cpu_arch, size(cp))
            ηs = partition(ηs, cpu_arch, size(ηp))

            atol = eps(eltype(global_underlying_grid))
            rtol = sqrt(eps(eltype(global_underlying_grid)))

            @test all(isapprox(up, us; atol, rtol))
            @test all(isapprox(vp, vs; atol, rtol))
            @test all(isapprox(wp, ws; atol, rtol))
            @test all(isapprox(cp, cs; atol, rtol))
            @test all(isapprox(ηp, ηs; atol, rtol))
        end
    end
end

