using Test
using MPI
using Oceananigans

using Oceananigans.BoundaryConditions: fill_halo_regions!

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD

# Right now just testing with 4 ranks!
mpi_ranks = MPI.Comm_size(comm)
@assert mpi_ranks == 4

#####
##### Multi architectures and rank connectivity
#####

function run_triply_periodic_rank_connectivity_tests_with_411_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))

    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test my_rank == index2rank(arch.my_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in y and z.
    @test isnothing(connectivity.south)
    @test isnothing(connectivity.north)
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+---+---+---+
    # | 0 | 1 | 2 | 3 |
    # +---+---+---+---+

    if my_rank == 0
        @test connectivity.east == 1
        @test connectivity.west == 3
    elseif my_rank == 1
        @test connectivity.east == 2
        @test connectivity.west == 0
    elseif my_rank == 2
        @test connectivity.east == 3
        @test connectivity.west == 1
    elseif my_rank == 3
        @test connectivity.east == 0
        @test connectivity.west == 2
    end

    return nothing
end

function run_triply_periodic_rank_connectivity_tests_with_141_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))

    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test my_rank == index2rank(arch.my_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in x and z.
    @test isnothing(connectivity.east)
    @test isnothing(connectivity.west)
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+
    # | 0 |
    # +---+
    # | 1 |
    # +---+
    # | 2 |
    # +---+
    # | 3 |
    # +---+

    if my_rank == 0
        @test connectivity.north == 1
        @test connectivity.south == 3
    elseif my_rank == 1
        @test connectivity.north == 2
        @test connectivity.south == 0
    elseif my_rank == 2
        @test connectivity.north == 3
        @test connectivity.south == 1
    elseif my_rank == 3
        @test connectivity.north == 0
        @test connectivity.south == 2
    end

    return nothing
end

function run_triply_periodic_rank_connectivity_tests_with_114_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))

    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test my_rank == index2rank(arch.my_index..., arch.ranks...)

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

    if my_rank == 0
        @test connectivity.top == 1
        @test connectivity.bottom == 3
    elseif my_rank == 1
        @test connectivity.top == 2
        @test connectivity.bottom == 0
    elseif my_rank == 2
        @test connectivity.top == 3
        @test connectivity.bottom == 1
    elseif my_rank == 3
        @test connectivity.top == 0
        @test connectivity.bottom == 2
    end

    return nothing
end

function run_triply_periodic_rank_connectivity_tests_with_221_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))

    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test my_rank == index2rank(arch.my_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in z.
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+---+
    # | 0 | 2 |
    # +---+---+
    # | 1 | 3 |
    # +---+---+

    if my_rank == 0
        @test connectivity.east == 2
        @test connectivity.west == 2
        @test connectivity.north == 1
        @test connectivity.south == 1
    elseif my_rank == 1
        @test connectivity.east == 3
        @test connectivity.west == 3
        @test connectivity.north == 0
        @test connectivity.south == 0
    elseif my_rank == 2
        @test connectivity.east == 0
        @test connectivity.west == 0
        @test connectivity.north == 3
        @test connectivity.south == 3
    elseif my_rank == 3
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

function run_triply_periodic_local_grid_tests_with_411_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_grid = dm.model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0.25*my_rank
    @test local_grid.xF[nx+1] == 0.25*(my_rank+1)
    @test local_grid.yF[1] == 0
    @test local_grid.yF[ny+1] == 2
    @test local_grid.zF[1] == -3
    @test local_grid.zF[nz+1] == 0

    return nothing
end

function run_triply_periodic_local_grid_tests_with_141_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_grid = dm.model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0
    @test local_grid.xF[nx+1] == 1
    @test local_grid.yF[1] == 0.5*my_rank
    @test local_grid.yF[ny+1] == 0.5*(my_rank+1)
    @test local_grid.zF[1] == -3
    @test local_grid.zF[nz+1] == 0

    return nothing
end

function run_triply_periodic_local_grid_tests_with_114_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_grid = dm.model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0
    @test local_grid.xF[nx+1] == 1
    @test local_grid.yF[1] == 0
    @test local_grid.yF[ny+1] == 2
    @test local_grid.zF[1] == -3 + 0.75*my_rank
    @test local_grid.zF[nz+1] == -3 + 0.75*(my_rank+1)

    return nothing
end

function run_triply_periodic_local_grid_tests_with_221_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    i, j, k = arch.my_index
    local_grid = dm.model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0.5*(i-1)
    @test local_grid.xF[nx+1] == 0.5*i
    @test local_grid.yF[1] == j-1
    @test local_grid.yF[ny+1] == j
    @test local_grid.zF[1] == -3
    @test local_grid.zF[nz+1] == 0

    return nothing
end

#####
##### Injection of halo communication BCs
#####

function run_triply_periodic_bc_injection_tests_with_411_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        fbcs = field.boundary_conditions
        @test fbcs.east isa HaloCommunicationBC
        @test fbcs.west isa HaloCommunicationBC
        @test !isa(fbcs.north, HaloCommunicationBC)
        @test !isa(fbcs.south, HaloCommunicationBC)
        @test !isa(fbcs.top, HaloCommunicationBC)
        @test !isa(fbcs.bottom, HaloCommunicationBC)
    end
end

function run_triply_periodic_bc_injection_tests_with_141_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        fbcs = field.boundary_conditions
        @test !isa(fbcs.east, HaloCommunicationBC)
        @test !isa(fbcs.west, HaloCommunicationBC)
        @test fbcs.north isa HaloCommunicationBC
        @test fbcs.south isa HaloCommunicationBC
        @test !isa(fbcs.top, HaloCommunicationBC)
        @test !isa(fbcs.bottom, HaloCommunicationBC)
    end
end

function run_triply_periodic_bc_injection_tests_with_114_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        fbcs = field.boundary_conditions
        @test !isa(fbcs.east, HaloCommunicationBC)
        @test !isa(fbcs.west, HaloCommunicationBC)
        @test !isa(fbcs.north, HaloCommunicationBC)
        @test !isa(fbcs.south, HaloCommunicationBC)
        @test fbcs.top isa HaloCommunicationBC
        @test fbcs.bottom isa HaloCommunicationBC
    end
end

function run_triply_periodic_bc_injection_tests_with_221_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        fbcs = field.boundary_conditions
        @test fbcs.east isa HaloCommunicationBC
        @test fbcs.west isa HaloCommunicationBC
        @test fbcs.north isa HaloCommunicationBC
        @test fbcs.south isa HaloCommunicationBC
        @test !isa(fbcs.top, HaloCommunicationBC)
        @test !isa(fbcs.bottom, HaloCommunicationBC)
    end
end

#####
##### Halo communication
#####

function run_triply_periodic_halo_communication_tests_with_411_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 6, 4), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        interior(field) .= arch.my_rank
        fill_halo_regions!(field, arch)

        @test all(east_halo(field) .== arch.connectivity.east)
        @test all(west_halo(field) .== arch.connectivity.west)

        @test all(interior(field) .== arch.my_rank)
        @test all(north_halo(field, include_corners=false) .== arch.my_rank)
        @test all(south_halo(field, include_corners=false) .== arch.my_rank)
        @test all(top_halo(field, include_corners=false) .== arch.my_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.my_rank)
    end

    return nothing
end

function run_triply_periodic_halo_communication_tests_with_141_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(3, 8, 2), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        interior(field) .= arch.my_rank
        fill_halo_regions!(field, arch)

        @test all(north_halo(field) .== arch.connectivity.north)
        @test all(south_halo(field) .== arch.connectivity.south)

        @test all(interior(field) .== arch.my_rank)
        @test all(east_halo(field, include_corners=false) .== arch.my_rank)
        @test all(west_halo(field, include_corners=false) .== arch.my_rank)
        @test all(top_halo(field, include_corners=false) .== arch.my_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.my_rank)
    end

    return nothing
end

function run_triply_periodic_halo_communication_tests_with_114_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(3, 5, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        interior(field) .= arch.my_rank
        fill_halo_regions!(field, arch)

        @test all(top_halo(field) .== arch.connectivity.top)
        @test all(bottom_halo(field) .== arch.connectivity.bottom)

        @test all(interior(field) .== arch.my_rank)
        @test all(east_halo(field, include_corners=false) .== arch.my_rank)
        @test all(west_halo(field, include_corners=false) .== arch.my_rank)
        @test all(north_halo(field, include_corners=false) .== arch.my_rank)
        @test all(south_halo(field, include_corners=false) .== arch.my_rank)
    end

    return nothing
end

function run_triply_periodic_halo_communication_tests_with_221_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularCartesianGrid(topology=topo, size=(8, 8, 3), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))
    dm = DistributedModel(architecture=arch, grid=full_grid)

    for field in fields(dm.model)
        interior(field) .= arch.my_rank
        fill_halo_regions!(field, arch)

        @test all(east_halo(field) .== arch.connectivity.east)
        @test all(west_halo(field) .== arch.connectivity.west)
        @test all(north_halo(field) .== arch.connectivity.north)
        @test all(south_halo(field) .== arch.connectivity.south)

        @test all(interior(field) .== arch.my_rank)
        @test all(top_halo(field, include_corners=false) .== arch.my_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.my_rank)
    end

    return nothing
end

#####
##### Run tests!
#####

@testset "Distributed MPI Oceananigans" begin
    @info "Testing distributed MPI Oceananigans..."

    @testset "Multi architectures rank connectivity" begin
        @info "  Testing multi architecture rank connectivity..."
        run_triply_periodic_rank_connectivity_tests_with_411_ranks()
        run_triply_periodic_rank_connectivity_tests_with_141_ranks()
        run_triply_periodic_rank_connectivity_tests_with_114_ranks()
        run_triply_periodic_rank_connectivity_tests_with_221_ranks()
    end

    @testset "Local grids for distributed models" begin
        @info "  Testing local grids for distributed models..."
        run_triply_periodic_local_grid_tests_with_411_ranks()
        run_triply_periodic_local_grid_tests_with_141_ranks()
        run_triply_periodic_local_grid_tests_with_114_ranks()
        run_triply_periodic_local_grid_tests_with_221_ranks()
    end

    @testset "Injection of halo communication BCs" begin
        @info "  Testing injection of halo communication BCs..."
        run_triply_periodic_bc_injection_tests_with_411_ranks()
        run_triply_periodic_bc_injection_tests_with_141_ranks()
        run_triply_periodic_bc_injection_tests_with_114_ranks()
        run_triply_periodic_bc_injection_tests_with_221_ranks()
    end

    # TODO: Test larger halos!
    @testset "Halo communication" begin
        @info "  Testing halo communication..."
        run_triply_periodic_halo_communication_tests_with_411_ranks()
        run_triply_periodic_halo_communication_tests_with_141_ranks()
        run_triply_periodic_halo_communication_tests_with_114_ranks()
        # run_triply_periodic_halo_communication_tests_with_221_ranks()
    end

    include("test_distributed_poisson_solvers.jl")
end

# MPI.Finalize()
# @test MPI.Finalized()
