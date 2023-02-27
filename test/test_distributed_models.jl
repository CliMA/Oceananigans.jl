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

MPI.Init()

# to initialize MPI.

using Oceananigans.BoundaryConditions: fill_halo_regions!, DCBC
using Oceananigans.Distributed: DistributedArch, index2rank, east_halo, west_halo, north_halo, south_halo, top_halo, bottom_halo

# Right now just testing with 4 ranks!
comm = MPI.COMM_WORLD
mpi_ranks = MPI.Comm_size(comm)
@assert mpi_ranks == 4

#####
##### Multi architectures and rank connectivity
#####

function test_triply_periodic_rank_connectivity_with_411_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(4, 1, 1), topology = topo, use_buffers = use_buffers)

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in y and z.
    @test isnothing(connectivity.south)
    @test isnothing(connectivity.north)
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+---+---+---+
    # | 0 | 1 | 2 | 3 |
    # +---+---+---+---+

    if local_rank == 0
        @test connectivity.east == 1
        @test connectivity.west == 3
    elseif local_rank == 1
        @test connectivity.east == 2
        @test connectivity.west == 0
    elseif local_rank == 2
        @test connectivity.east == 3
        @test connectivity.west == 1
    elseif local_rank == 3
        @test connectivity.east == 0
        @test connectivity.west == 2
    end

    return nothing
end

function test_triply_periodic_rank_connectivity_with_141_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(1, 4, 1), topology = topo, use_buffers = use_buffers)

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in x and z.
    @test isnothing(connectivity.east)
    @test isnothing(connectivity.west)
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+
    # | 3 |
    # +---+
    # | 2 |
    # +---+
    # | 1 |
    # +---+
    # | 0 |
    # +---+

    if local_rank == 0
        @test connectivity.north == 1
        @test connectivity.south == 3
    elseif local_rank == 1
        @test connectivity.north == 2
        @test connectivity.south == 0
    elseif local_rank == 2
        @test connectivity.north == 3
        @test connectivity.south == 1
    elseif local_rank == 3
        @test connectivity.north == 0
        @test connectivity.south == 2
    end

    return nothing
end

function test_triply_periodic_rank_connectivity_with_114_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(1, 1, 4), topology = topo, use_buffers = use_buffers)

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in x and y.
    @test isnothing(connectivity.east)
    @test isnothing(connectivity.west)
    @test isnothing(connectivity.north)
    @test isnothing(connectivity.south)

    #   /---/
    #  / 3 /
    # /---/
    #   /---/
    #  / 2 /
    # /---/
    #   /---/
    #  / 1 /
    # /---/
    #   /---/
    #  / 0 /
    # /---/

    if local_rank == 0
        @test connectivity.top == 1
        @test connectivity.bottom == 3
    elseif local_rank == 1
        @test connectivity.top == 2
        @test connectivity.bottom == 0
    elseif local_rank == 2
        @test connectivity.top == 3
        @test connectivity.bottom == 1
    elseif local_rank == 3
        @test connectivity.top == 0
        @test connectivity.bottom == 2
    end

    return nothing
end

function test_triply_periodic_rank_connectivity_with_221_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(2, 2, 1), topology = topo, use_buffers = use_buffers)

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in z.
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+---+
    # | 0 | 2 |
    # +---+---+
    # | 1 | 3 |
    # +---+---+

    if local_rank == 0
        @test connectivity.east == 2
        @test connectivity.west == 2
        @test connectivity.north == 1
        @test connectivity.south == 1
    elseif local_rank == 1
        @test connectivity.east == 3
        @test connectivity.west == 3
        @test connectivity.north == 0
        @test connectivity.south == 0
    elseif local_rank == 2
        @test connectivity.east == 0
        @test connectivity.west == 0
        @test connectivity.north == 3
        @test connectivity.south == 3
    elseif local_rank == 3
        @test connectivity.east == 1
        @test connectivity.west == 1
        @test connectivity.north == 2
        @test connectivity.south == 2
    end

    return nothing
end

#####
##### Local grids for distributed models
#####

function test_triply_periodic_local_grid_with_411_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(4, 1, 1), topology = topo, use_buffers = use_buffers)
    local_grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nx, ny, nz = size(local_grid)

    @test local_grid.xᶠᵃᵃ[1] == 0.25*local_rank
    @test local_grid.xᶠᵃᵃ[nx+1] == 0.25*(local_rank+1)
    @test local_grid.yᵃᶠᵃ[1] == 0
    @test local_grid.yᵃᶠᵃ[ny+1] == 2
    @test local_grid.zᵃᵃᶠ[1] == -3
    @test local_grid.zᵃᵃᶠ[nz+1] == 0

    return nothing
end

function test_triply_periodic_local_grid_with_141_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(1, 4, 1), topology = topo, use_buffers = use_buffers)
    local_grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nx, ny, nz = size(local_grid)

    @test local_grid.xᶠᵃᵃ[1] == 0
    @test local_grid.xᶠᵃᵃ[nx+1] == 1
    @test local_grid.yᵃᶠᵃ[1] == 0.5*local_rank
    @test local_grid.yᵃᶠᵃ[ny+1] == 0.5*(local_rank+1)
    @test local_grid.zᵃᵃᶠ[1] == -3
    @test local_grid.zᵃᵃᶠ[nz+1] == 0

    return nothing
end

function test_triply_periodic_local_grid_with_114_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(1, 1, 4), topology = topo, use_buffers = use_buffers)
    local_grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    
    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nx, ny, nz = size(local_grid)

    @test local_grid.xᶠᵃᵃ[1] == 0
    @test local_grid.xᶠᵃᵃ[nx+1] == 1
    @test local_grid.yᵃᶠᵃ[1] == 0
    @test local_grid.yᵃᶠᵃ[ny+1] == 2
    @test local_grid.zᵃᵃᶠ[1] == -3 + 0.75*local_rank
    @test local_grid.zᵃᵃᶠ[nz+1] == -3 + 0.75*(local_rank+1)

    return nothing
end

function test_triply_periodic_local_grid_with_221_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(2, 2, 1), topology = topo, use_buffers = use_buffers)
    local_grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    
    i, j, k = arch.local_index
    nx, ny, nz = size(local_grid)

    @test local_grid.xᶠᵃᵃ[1] == 0.5*(i-1)
    @test local_grid.xᶠᵃᵃ[nx+1] == 0.5*i
    @test local_grid.yᵃᶠᵃ[1] == j-1
    @test local_grid.yᵃᶠᵃ[ny+1] == j
    @test local_grid.zᵃᵃᶠ[1] == -3
    @test local_grid.zᵃᵃᶠ[nz+1] == 0

    return nothing
end

#####
##### Injection of halo communication BCs
#####
##### TODO: use Field constructor for these tests rather than NonhydrostaticModel.
#####

function test_triply_periodic_bc_injection_with_411_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(4, 1, 1), topology=topo, use_buffers = use_buffers)
    grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid=grid)

    for field in merge(fields(model))
        fbcs = field.boundary_conditions
        @test fbcs.east isa DCBC
        @test fbcs.west isa DCBC
        @test !isa(fbcs.north, DCBC)
        @test !isa(fbcs.south, DCBC)
        @test !isa(fbcs.top, DCBC)
        @test !isa(fbcs.bottom, DCBC)
    end
end

function test_triply_periodic_bc_injection_with_141_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(1, 4, 1), use_buffers = use_buffers)
    grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid=grid)

    for field in merge(fields(model))
        fbcs = field.boundary_conditions
        @test !isa(fbcs.east, DCBC)
        @test !isa(fbcs.west, DCBC)
        @test fbcs.north isa DCBC
        @test fbcs.south isa DCBC
        @test !isa(fbcs.top, DCBC)
        @test !isa(fbcs.bottom, DCBC)
    end
end

function test_triply_periodic_bc_injection_with_114_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(1, 1, 4), use_buffers = use_buffers)
    grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid=grid)

    for field in merge(fields(model))
        fbcs = field.boundary_conditions
        @test !isa(fbcs.east, DCBC)
        @test !isa(fbcs.west, DCBC)
        @test !isa(fbcs.north, DCBC)
        @test !isa(fbcs.south, DCBC)
        @test fbcs.top isa DCBC
        @test fbcs.bottom isa DCBC
    end
end

function test_triply_periodic_bc_injection_with_221_ranks(child_arch, use_buffers)
    topo = (Periodic, Periodic, Periodic)
    arch = DistributedArch(child_arch, ranks=(2, 2, 1), use_buffers = use_buffers)
    grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid=grid)

    for field in merge(fields(model))
        fbcs = field.boundary_conditions
        @test fbcs.east isa DCBC
        @test fbcs.west isa DCBC
        @test fbcs.north isa DCBC
        @test fbcs.south isa DCBC
        @test !isa(fbcs.top, DCBC)
        @test !isa(fbcs.bottom, DCBC)
    end
end

#####
##### Halo communication
#####

function test_triply_periodic_halo_communication_with_411_ranks(halo, child_arch)
    topo = (Periodic, Periodic, Periodic)
    for use_buffers in (true , false)
        arch = DistributedArch(child_arch; ranks=(4, 1, 1), use_buffers = use_buffers)
        grid = RectilinearGrid(arch, topology=topo, size=(16, 6, 4), extent=(1, 2, 3), halo=halo)
        model = NonhydrostaticModel(grid=grid)

        for field in merge(fields(model))
            interior(field) .= arch.local_rank
            fill_halo_regions!(field)

            @test all(east_halo(field, include_corners=false) .== arch.connectivity.east)
            @test all(west_halo(field, include_corners=false) .== arch.connectivity.west)

            @test all(interior(field) .== arch.local_rank)
            @test all(north_halo(field, include_corners=false) .== arch.local_rank)
            @test all(south_halo(field, include_corners=false) .== arch.local_rank)
            @test all(top_halo(field, include_corners=false) .== arch.local_rank)
            @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
        end
    end


    return nothing
end

function test_triply_periodic_halo_communication_with_141_ranks(halo, child_arch)
    topo  = (Periodic, Periodic, Periodic)
    for use_buffers in (true , false)
        arch = DistributedArch(child_arch; ranks=(1, 4, 1), use_buffers = use_buffers)
        grid  = RectilinearGrid(arch, topology=topo, size=(4, 16, 4), extent=(1, 2, 3), halo=halo)
        model = NonhydrostaticModel(grid=grid)

        for field in merge(fields(model), model.pressures)
            interior(field) .= arch.local_rank
            fill_halo_regions!(field)

            @test all(north_halo(field, include_corners=false) .== arch.connectivity.north)
            @test all(south_halo(field, include_corners=false) .== arch.connectivity.south)

            @test all(interior(field) .== arch.local_rank)
            @test all(east_halo(field, include_corners=false) .== arch.local_rank)
            @test all(west_halo(field, include_corners=false) .== arch.local_rank)
            @test all(top_halo(field, include_corners=false) .== arch.local_rank)
            @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
        end
    end
    return nothing
end

function test_triply_periodic_halo_communication_with_114_ranks(halo, child_arch)
    topo = (Periodic, Periodic, Periodic)
    for use_buffers in (true , false)
        arch = DistributedArch(child_arch; ranks=(1, 4, 1), use_buffers)
        grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 16), extent=(1, 2, 3), halo=halo)
        model = NonhydrostaticModel(grid=grid)

        for field in merge(fields(model))
            interior(field) .= arch.local_rank
            fill_halo_regions!(field)

            @test all(top_halo(field, include_corners=false) .== arch.connectivity.top)
            @test all(bottom_halo(field, include_corners=false) .== arch.connectivity.bottom)

            @test all(interior(field) .== arch.local_rank)
            @test all(east_halo(field, include_corners=false) .== arch.local_rank)
            @test all(west_halo(field, include_corners=false) .== arch.local_rank)
            @test all(north_halo(field, include_corners=false) .== arch.local_rank)
            @test all(south_halo(field, include_corners=false) .== arch.local_rank)
        end
    end

    return nothing
end

function test_triply_periodic_halo_communication_with_221_ranks(halo, child_arch)
    topo = (Periodic, Periodic, Periodic)
    for use_buffers in (true , false)
        arch = DistributedArch(child_arch; ranks=(2, 2, 1), use_buffers, devices = (0, 0, 0, 0))
        grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 3), extent=(1, 2, 3), halo=halo)
        model = NonhydrostaticModel(grid=grid)

        for field in merge(fields(model))
            interior(field) .= arch.local_rank
            fill_halo_regions!(field)

            @test all(east_halo(field, include_corners=false) .== arch.connectivity.east)
            @test all(west_halo(field, include_corners=false) .== arch.connectivity.west)
            @test all(north_halo(field, include_corners=false) .== arch.connectivity.north)
            @test all(south_halo(field, include_corners=false) .== arch.connectivity.south)

            @test all(interior(field) .== arch.local_rank)
            @test all(top_halo(field, include_corners=false) .== arch.local_rank)
            @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
        end
    end

    return nothing
end

#####
##### Run tests!
#####

@testset "Distributed MPI Oceananigans" begin

    for child_arch ∈ archs

        @info "Testing distributed MPI Oceananigans..."

        # We don't support distributing _anything_ in the vertical,
        # so these tests are commented out below (and maybe should be removed
        # in the future).

        @testset "Multi architectures rank connectivity" begin
            @info "  Testing multi architecture rank connectivity on $child_arch..."
            test_triply_periodic_rank_connectivity_with_411_ranks(child_arch, child_arch isa GPU)
            test_triply_periodic_rank_connectivity_with_141_ranks(child_arch, child_arch isa GPU)
            # test_triply_periodic_rank_connectivity_with_114_ranks(child_arch, child_arch isa GPU)
            test_triply_periodic_rank_connectivity_with_221_ranks(child_arch, child_arch isa GPU)
        end

        @testset "Local grids for distributed models" begin
            @info "  Testing local grids for distributed models on $child_arch..."
            test_triply_periodic_local_grid_with_411_ranks(child_arch, child_arch isa GPU)
            test_triply_periodic_local_grid_with_141_ranks(child_arch, child_arch isa GPU)
            # test_triply_periodic_local_grid_with_114_ranks(child_arch, child_arch isa GPU)
            test_triply_periodic_local_grid_with_221_ranks(child_arch, child_arch isa GPU)
        end

        @testset "Injection of halo communication BCs" begin
            @info "  Testing injection of halo communication BCs on $child_arch..."
            test_triply_periodic_bc_injection_with_411_ranks(child_arch, child_arch isa GPU)
            test_triply_periodic_bc_injection_with_141_ranks(child_arch, child_arch isa GPU)
            # test_triply_periodic_bc_injection_with_114_ranks(child_arch, child_arch isa GPU)
            test_triply_periodic_bc_injection_with_221_ranks(child_arch, child_arch isa GPU)
        end

        @testset "Halo communication" begin
            @info "  Testing halo communication on $child_arch..."
            for H in 1:3
                test_triply_periodic_halo_communication_with_411_ranks((H, H, H), child_arch)
                test_triply_periodic_halo_communication_with_141_ranks((H, H, H), child_arch)
                # test_triply_periodic_halo_communication_with_114_ranks((H, H, H), child_arch)
                test_triply_periodic_halo_communication_with_221_ranks((H, H, H), child_arch)
            end
        end

        # Only test on CPU because we do not have a GPU pressure solver yet
        @testset "Time stepping NonhydrostaticModel on $child_arch" begin
            for ranks in [(1, 4, 1), (2, 2, 1), (4, 1, 1)]
                @info "Time-stepping a distributed NonhydrostaticModel with ranks $ranks..."
                topo = (Periodic, Periodic, Periodic)
                arch = DistributedArch(; ranks)
                grid = RectilinearGrid(arch, topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
                model = NonhydrostaticModel(; grid)

                time_step!(model, 1)
                @test model isa NonhydrostaticModel
                @test model.clock.time ≈ 1

                simulation = Simulation(model, Δt=1, stop_iteration=2)
                run!(simulation)
                @test model isa NonhydrostaticModel
                @test model.clock.time ≈ 2
            end
        end

        @testset "Time stepping ShallowWaterModel on $child_arch" begin
            topo = (Periodic, Periodic, Flat)
            use_buffers = child_arch isa GPU ? true : false
            arch = DistributedArch(child_arch; ranks=(1, 4, 1), topology = topo, use_buffers, devices = (0, 0, 0, 0))
            grid = RectilinearGrid(arch, topology=topo, size=(8, 8), extent=(1, 2), halo=(3, 3))
            model = ShallowWaterModel(; momentum_advection=nothing, mass_advection=nothing, tracer_advection=nothing, grid, gravitational_acceleration=1)

            set!(model, h=1)
            time_step!(model, 1)
            @test model isa ShallowWaterModel
            @test model.clock.time ≈ 1

            simulation = Simulation(model, Δt=1, stop_iteration=2)
            run!(simulation)
            @test model isa ShallowWaterModel
            @test model.clock.time ≈ 2
        end
    end
end

