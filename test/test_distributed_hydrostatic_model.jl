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

using Oceananigans: prognostic_fields
using Oceananigans.Operators: hack_cosd
using Oceananigans.DistributedComputations: ranks, partition, all_reduce, cpu_architecture, reconstruct_global_grid, synchronized
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity

function Δ_min(grid)
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

function test_model_equality(test_model, true_model)
    @allowscalar begin
        test_model_fields = prognostic_fields(test_model)
        true_model_fields = prognostic_fields(true_model)
        field_names = keys(test_model_fields)

        for name in field_names
            @test all(test_model_fields[name].data .≈ true_model_fields[name].data)

            if test_model.timestepper isa QuasiAdamsBashforth2TimeStepper
                if name ∈ keys(test_model.timestepper.Gⁿ)
                    @test all(test_model.timestepper.Gⁿ[name].data .≈ true_model.timestepper.Gⁿ[name].data)
                    @test all(test_model.timestepper.G⁻[name].data .≈ true_model.timestepper.G⁻[name].data)
                end
            end
        end
    end

    return nothing
end

@inline Gaussian(x, y, L) = exp(-(x^2 + y^2) / L^2)

function rotation_with_shear_test(grid, closure=nothing)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 8, gravitational_acceleration = 1)
    coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1)

    tracers = if closure isa CATKEVerticalDiffusivity
        (:c, :b, :e)
    else
        (:c, :b)
    end

    model = HydrostaticFreeSurfaceModel(; grid, closure, tracers, free_surface,
                                        momentum_advection = WENOVectorInvariant(order=3),
                                        coriolis = coriolis,
                                        tracer_advection = WENO(order=3),
                                        buoyancy = BuoyancyTracer())

    g = model.free_surface.gravitational_acceleration
    R = grid.radius
    Ω = model.coriolis.rotation_rate

    # Add some shear on the velocity field
    uᵢ(λ, φ, z) = 0.1 * cosd(φ) * sind(λ) + 0.05 * z
    ηᵢ(λ, φ, z) = (R * Ω * 0.1 + 0.1^2 / 2) * sind(φ)^2 / g * sind(λ)

    # Gaussian leads to values with O(1e-60); too small for reproducibility.
    # We cap it at 0.1
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

            if arch.local_rank == 0
                @info "  Testing CATKE with $(ranks(arch)) ranks"
            end

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

    @testset "Distributed checkpointing" begin
        # Create and run "true model"
        Nx, Ny, Nz = 16, 16, 4
        Lx, Ly, Lz = 1, 1, 1

        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(-10, 10), y=(-10, 10), z=(-1, 0))
        free_surface = SplitExplicitFreeSurface(grid; substeps=8, gravitational_acceleration=1)
        closure = ScalarDiffusivity(ν=1e-2)
        true_model = HydrostaticFreeSurfaceModel(; grid, free_surface, closure)
        test_model = deepcopy(true_model)

        ηᵢ(x, y, z) = 1e-1 * exp(-x^2 - y^2)
        ϵᵢ(x, y, z) = 1e-6 * randn()
        set!(true_model, η=ηᵢ, u=ϵᵢ, v=ϵᵢ)

        Δt = 1e-6
        true_simulation = Simulation(true_model, Δt=Δt, stop_iteration=5)
        checkpointer = Checkpointer(true_model, schedule=IterationInterval(5), overwrite_existing=true)
        push!(true_simulation.output_writers, checkpointer)
        run!(true_simulation) # for 5 iterations
        checkpointed_model = deepcopy(true_simulation.model)

        true_simulation.stop_iteration = 9
        run!(true_simulation) # for 4 more iterations

        # Let's wait until all ranks complete the simulation!
        Oceananigans.DistributedComputations.barrier(arch)

        #####
        ##### Test `set!(model, checkpoint_file)`
        #####

        rank = arch.local_rank
        set!(test_model, "checkpoint_rank$(rank)_iteration5.jld2")
        
        @test test_model.clock.iteration == checkpointed_model.clock.iteration
        @test test_model.clock.time == checkpointed_model.clock.time
        test_model_equality(test_model, checkpointed_model)

        # This only applies to QuasiAdamsBashforthTimeStepper:
        @test test_model.clock.last_Δt == checkpointed_model.clock.last_Δt

        #####
        ##### Test pickup from explicit checkpoint path
        #####

        test_simulation = Simulation(test_model, Δt=Δt, stop_iteration=9)

        # Pickup from explicit checkpoint path
        run!(test_simulation, pickup="checkpoint_rank$(rank)_iteration0.jld2")

        @info "Testing model equality when running with pickup=checkpoint_iteration0.jld2."
        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        Oceananigans.DistributedComputations.barrier(arch)
        run!(test_simulation, pickup="checkpoint_rank$(rank)_iteration5.jld2")
        @info "Testing model equality when running with pickup=checkpoint_iteration5.jld2."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        #####
        ##### Test `run!(sim, pickup=true)
        #####

        # Pickup using existing checkpointer
        test_simulation.output_writers[:checkpointer] =
            Checkpointer(test_model, schedule=IterationInterval(5), overwrite_existing=true)

        Oceananigans.DistributedComputations.barrier(arch)
        run!(test_simulation, pickup=true)
        @info "    Testing model equality when running with pickup=true."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        Oceananigans.DistributedComputations.barrier(arch)
        run!(test_simulation, pickup=0)
        @info "    Testing model equality when running with pickup=0."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        Oceananigans.DistributedComputations.barrier(arch)
        run!(test_simulation, pickup=5)
        @info "    Testing model equality when running with pickup=5."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)
        Oceananigans.DistributedComputations.barrier(arch)
        
        for iteration in (0, 5)
            rm("checkpoint_rank$(rank)_iteration$(iteration).jld2", force=true)
        end
    end
end
