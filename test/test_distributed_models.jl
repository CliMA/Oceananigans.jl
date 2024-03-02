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
using Oceananigans.DistributedComputations: Distributed, index2rank
using Oceananigans.Fields: AbstractField
using Oceananigans.Grids:
    halo_size,
    interior_indices,
    left_halo_indices, right_halo_indices,
    underlying_left_halo_indices, underlying_right_halo_indices

#####
##### Viewing halos
#####

instantiate(T::Type) = T()
instantiate(t) = t

west_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, left_halo_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx, f.grid.Hx), :, :) :
                      view(f.data, left_halo_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx, f.grid.Hx),
                                   interior_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny),
                                   interior_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz))

east_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, right_halo_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx, f.grid.Hx), :, :) :
                      view(f.data, right_halo_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx, f.grid.Hx),
                                   interior_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny),
                                   interior_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz))

south_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, :, left_halo_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny, f.grid.Hy), :) :
                      view(f.data, interior_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx),
                                   left_halo_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny, f.grid.Hy),
                                   interior_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz))

north_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, :, right_halo_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny, f.grid.Hy), :) :
                      view(f.data, interior_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx),
                                   right_halo_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny, f.grid.Hy),
                                   interior_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz))
                        
bottom_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
include_corners ? view(f.data, :, :, left_halo_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz, f.grid.Hz)) :
                  view(f.data, interior_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx),
                               interior_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny),
                               left_halo_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz, f.grid.Hz))

top_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
include_corners ? view(f.data, :, :, right_halo_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz, f.grid.Hz)) :
                  view(f.data, interior_indices(instantiate(LX), instantiate(topology(f, 1)), f.grid.Nx),
                               interior_indices(instantiate(LY), instantiate(topology(f, 2)), f.grid.Ny),
                               right_halo_indices(instantiate(LZ), instantiate(topology(f, 3)), f.grid.Nz, f.grid.Hz))


function southwest_halo(f::AbstractField) 
    Nx, Ny, _ = size(f.grid)
    Hx, Hy, _ = halo_size(f.grid)
    return view(parent(f), 1:Hx, 1:Hy, :)
end

function southeast_halo(f::AbstractField) 
    Nx, Ny, _ = size(f.grid)
    Hx, Hy, _ = halo_size(f.grid)
    return view(parent(f), Nx+Hx+1:Nx+2Hx, 1:Hy, :)
end

function northeast_halo(f::AbstractField) 
    Nx, Ny, _ = size(f.grid)
    Hx, Hy, _ = halo_size(f.grid)
    return view(parent(f), Nx+Hx+1:Nx+2Hx, Ny+Hy+1:Ny+2Hy, :)
end

function northwest_halo(f::AbstractField) 
    Nx, Ny, _ = size(f.grid)
    Hx, Hy, _ = halo_size(f.grid)
    return view(parent(f), 1:Hx, Ny+Hy+1:Ny+2Hy, :)
end

# Right now just testing with 4 ranks!
comm = MPI.COMM_WORLD
mpi_ranks = MPI.Comm_size(comm)
@assert mpi_ranks == 4

#####
##### Multi architectures and rank connectivity
#####

function test_triply_periodic_rank_connectivity_with_411_ranks()
    arch = Distributed(CPU(), partition=Partition(4))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in y.
    @test isnothing(connectivity.south)
    @test isnothing(connectivity.north)

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

function test_triply_periodic_rank_connectivity_with_141_ranks()
    arch = Distributed(CPU(), partition=Partition(1, 4))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in x.
    @test isnothing(connectivity.east)
    @test isnothing(connectivity.west)

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

function test_triply_periodic_rank_connectivity_with_221_ranks()
    arch = Distributed(CPU(), partition=Partition(2, 2))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity
    
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

function test_triply_periodic_local_grid_with_411_ranks()
    arch = Distributed(CPU(), partition=Partition(4))
    local_grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3))

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

function test_triply_periodic_local_grid_with_141_ranks()
    arch = Distributed(CPU(), partition=Partition(1, 4))
    local_grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3))

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

function test_triply_periodic_local_grid_with_221_ranks()
    arch = Distributed(CPU(), partition=Partition(2, 2))
    local_grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3))
    
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

function test_triply_periodic_bc_injection_with_411_ranks()
    arch = Distributed(partition=Partition(4))
    grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3))
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

function test_triply_periodic_bc_injection_with_141_ranks()
    arch = Distributed(partition=Partition(1, 4))
    grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3))
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

function test_triply_periodic_bc_injection_with_221_ranks()
    arch = Distributed(partition=Partition(2, 2))
    grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3))
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
    arch = Distributed(child_arch; partition=Partition(4))
    grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3), halo=halo)
    model = NonhydrostaticModel(grid=grid)

    for field in merge(fields(model))
        fill!(field, arch.local_rank)
        fill_halo_regions!(field)

        @test all(east_halo(field, include_corners=false) .== arch.connectivity.east)
        @test all(west_halo(field, include_corners=false) .== arch.connectivity.west)

        @test all(interior(field) .== arch.local_rank)
        @test all(north_halo(field, include_corners=false) .== arch.local_rank)
        @test all(south_halo(field, include_corners=false) .== arch.local_rank)
        @test all(top_halo(field, include_corners=false) .== arch.local_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
    end

    return nothing
end

function test_triply_periodic_halo_communication_with_141_ranks(halo, child_arch)
    arch = Distributed(child_arch; partition=Partition(1, 4))
    grid  = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3), halo=halo)
    model = NonhydrostaticModel(grid=grid)

    for field in merge(fields(model), model.pressures)
        fill!(field, arch.local_rank)
        fill_halo_regions!(field)

        @test all(north_halo(field, include_corners=false) .== arch.connectivity.north)
        @test all(south_halo(field, include_corners=false) .== arch.connectivity.south)

        @test all(interior(field) .== arch.local_rank)
        @test all(east_halo(field, include_corners=false) .== arch.local_rank)
        @test all(west_halo(field, include_corners=false) .== arch.local_rank)
        @test all(top_halo(field, include_corners=false) .== arch.local_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
    end

    return nothing
end

function test_triply_periodic_halo_communication_with_221_ranks(halo, child_arch)
    arch = Distributed(child_arch; partition=Partition(2, 2))
    grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 3), extent=(1, 2, 3), halo=halo)
    model = NonhydrostaticModel(grid=grid)

    for field in merge(fields(model))
        fill!(field, arch.local_rank)
        fill_halo_regions!(field)

        @test all(interior(field) .== arch.local_rank)

        @test all(east_halo(field, include_corners=false)  .== arch.connectivity.east)
        @test all(west_halo(field, include_corners=false)  .== arch.connectivity.west)
        @test all(north_halo(field, include_corners=false) .== arch.connectivity.north)
        @test all(south_halo(field, include_corners=false) .== arch.connectivity.south)

        @test all(top_halo(field, include_corners=false)    .== arch.local_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
        @test all(southwest_halo(field) .== arch.connectivity.southwest) 
        @test all(southeast_halo(field) .== arch.connectivity.southeast) 
        @test all(northwest_halo(field) .== arch.connectivity.northwest) 
        @test all(northeast_halo(field) .== arch.connectivity.northeast) 
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
        test_triply_periodic_rank_connectivity_with_411_ranks()
        test_triply_periodic_rank_connectivity_with_141_ranks()
        test_triply_periodic_rank_connectivity_with_221_ranks()
    end

    @testset "Local grids for distributed models" begin
        @info "  Testing local grids for distributed models..."
        test_triply_periodic_local_grid_with_411_ranks()
        test_triply_periodic_local_grid_with_141_ranks()
        test_triply_periodic_local_grid_with_221_ranks()
    end

    @testset "Injection of halo communication BCs" begin
        @info "  Testing injection of halo communication BCs..."
        test_triply_periodic_bc_injection_with_411_ranks()
        test_triply_periodic_bc_injection_with_141_ranks()
        test_triply_periodic_bc_injection_with_221_ranks()
    end

    @testset "Halo communication" begin
        @info "  Testing halo communication..."
        for child_arch in archs
            for H in 1:3
                test_triply_periodic_halo_communication_with_411_ranks((H, H, H), child_arch)
                test_triply_periodic_halo_communication_with_141_ranks((H, H, H), child_arch)
                test_triply_periodic_halo_communication_with_221_ranks((H, H, H), child_arch)
            end
        end
    end

    # Only test on CPU because we do not have a GPU pressure solver yet
    @testset "Time stepping NonhydrostaticModel" begin
        if CPU() ∈ archs 
            for partition in [Partition(1, 4), Partition(2, 2), Partition(4, 1)]
                @info "Time-stepping a distributed NonhydrostaticModel with partition $partition..."
                arch = Distributed(; partition)
                grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Periodic), size=(8, 8, 8), extent=(1, 2, 3))
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
    end

    @testset "Time stepping ShallowWaterModel" begin
        for child_arch in archs
            arch = Distributed(child_arch; partition=Partition(1, 4))
            grid = RectilinearGrid(arch, topology=(Periodic, Periodic, Flat), size=(8, 8), extent=(1, 2), halo=(3, 3))
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
