import MPI

using Oceananigans
using Oceananigans: BCType, PBC
using Oceananigans.Grids: validate_tupled_argument

#####
##### Convinient aliases
#####

const PeriodicBC = PBC

#####
##### Converting between index and MPI rank taking k as the fast index
#####

@inline index2rank(i, j, k, Rx, Ry, Rz) = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)

@inline function rank2index(r, Rx, Ry, Rz)
    i = div(r, Ry*Rz)
    r -= i*Ry*Rz
    j = div(r, Rz)
    k = mod(r, Rz)
    return i+1, j+1, k+1
end

#####
##### Connectivity graph
#####

const Connectivity = NamedTuple{(:east, :west, :north, :south, :top, :bottom)}

function increment_index(i, R, bc)
    if i+1 > R
        if bc isa PeriodicBC
            return 1
        else
            return nothing
        end
    else
        return i+1
    end
end

function decrement_index(i, R, bc)
    if i-1 < 1
        if bc isa PeriodicBC
            return R
        else
            return nothing
        end
    else
        return i-1
    end
end

function construct_connectivity(index, ranks, bcs)
    i, j, k = index
    Rx, Ry, Rz = ranks

    i_east  = increment_index(i, Rx, bcs.x.right)
    i_west  = decrement_index(i, Rx, bcs.x.left)
    j_north = increment_index(j, Ry, bcs.y.north)
    j_south = decrement_index(j, Ry, bcs.y.south)
    k_top   = increment_index(k, Rz, bcs.z.top)
    k_bot   = decrement_index(k, Rz, bcs.z.bottom)

    r_east  = isnothing(i_east)  ? nothing : index2rank(i_east, j, k, Rx, Ry, Rz)
    r_west  = isnothing(i_west)  ? nothing : index2rank(i_west, j, k, Rx, Ry, Rz)
    r_north = isnothing(j_north) ? nothing : index2rank(i, j_north, k, Rx, Ry, Rz)
    r_south = isnothing(j_south) ? nothing : index2rank(i, j_south, k, Rx, Ry, Rz)
    r_top   = isnothing(k_top)   ? nothing : index2rank(i, j, k_top, Rx, Ry, Rz)
    r_bot   = isnothing(k_bot)   ? nothing : index2rank(i, j, k_bot, Rx, Ry, Rz)

    return (east=r_east, west=r_west, north=r_north,
            south=r_south, top=r_top, bottom=r_bot)
end

#####
##### Halo communication boundary condition
#####

struct HaloCommunication <: BCType end
const HaloCommunicationBC = BoundaryCondition{<:HaloCommunication}

const HaloCommunicationDetails = NamedTuple{(:rank_from, :rank_to)}
HaloCommunicationDetails(; rank_from, rank_to) = HaloCommunicationDetails((rank_from, rank_to))

function inject_halo_communication_boundary_conditions(boundary_conditions, my_rank, connectivity)
    new_field_bcs = []

    for field_bcs in boundary_conditions
        rank_east = connectivity.east
        rank_west = connectivity.west
        rank_north = connectivity.north
        rank_south = connectivity.south
        rank_top = connectivity.top
        rank_bottom = connectivity.bottom

        east_comm_bc_details = HaloCommunicationDetails(rank_from=my_rank, rank_to=connectivity.east)
        west_comm_bc_details = HaloCommunicationDetails(rank_from=my_rank, rank_to=connectivity.west)
        north_comm_bc_details = HaloCommunicationDetails(rank_from=my_rank, rank_to=connectivity.north)
        south_comm_bc_details = HaloCommunicationDetails(rank_from=my_rank, rank_to=connectivity.south)
        top_comm_bc_details = HaloCommunicationDetails(rank_from=my_rank, rank_to=connectivity.top)
        bottom_comm_bc_details = HaloCommunicationDetails(rank_from=my_rank, rank_to=connectivity.bottom)

        east_comm_bc = BoundaryCondition(HaloCommunication, east_comm_bc_details)
        west_comm_bc = BoundaryCondition(HaloCommunication, west_comm_bc_details)
        north_comm_bc = BoundaryCondition(HaloCommunication, north_comm_bc_details)
        south_comm_bc = BoundaryCondition(HaloCommunication, south_comm_bc_details)
        top_comm_bc = BoundaryCondition(HaloCommunication, top_comm_bc_details)
        bottom_comm_bc = BoundaryCondition(HaloCommunication, bottom_comm_bc_details)

        x_bcs = CoordinateBoundaryConditions(isnothing(rank_west) ? field_bcs.x.left : west_comm_bc,
                                             isnothing(rank_east) ? field_bcs.x.right : east_comm_bc)

        y_bcs = CoordinateBoundaryConditions(isnothing(rank_south) ? field_bcs.y.south : south_comm_bc,
                                             isnothing(rank_north) ? field_bcs.y.north : north_comm_bc)

        z_bcs = CoordinateBoundaryConditions(isnothing(rank_bottom) ? field_bcs.z.bottom : bottom_comm_bc,
                                             isnothing(rank_top) ? field_bcs.z.top : top_comm_bc)

        push!(new_field_bcs, FieldBoundaryConditions(x_bcs, y_bcs, z_bcs))
    end

    return NamedTuple{propertynames(boundary_conditions)}(Tuple(new_field_bcs))
end

#####
##### Distributed model struct and constructor
#####

struct DistributedModel{I, A, R, G}
                 index :: I
                 ranks :: R
                 model :: A
          connectivity :: G
end

"""
    DistributedModel(size, x, y, z, ranks, model_kwargs...)

size: Number of total grid points.
x, y, z: Left and right endpoints for each dimension.
ranks: Number of ranks in each dimension.
model_kwargs: Passed to `Model` constructor.
"""
function DistributedModel(; size, x, y, z, ranks, boundary_conditions, model_kwargs...)
    validate_tupled_argument(ranks, Int, "size")
    validate_tupled_argument(ranks, Int, "ranks")

    Nx, Ny, Nz = size

    # Pull out left and right endpoints for full model.
    xL, xR = x
    yL, yR = y
    zL, zR = z
    Lx, Ly, Lz = xR-xL, yR-yL, zR-zL

    Rx, Ry, Rz = ranks
    total_ranks = Rx*Ry*Rz

    comm = MPI.COMM_WORLD

    mpi_ranks = MPI.Comm_size(comm)
    my_rank   = MPI.Comm_rank(comm)

    if total_ranks != mpi_ranks
        throw(ArgumentError("ranks=($Rx, $Ry, $Rz) [$total_ranks total] inconsistent " *
                            "with number of MPI ranks: $mpi_ranks. Exiting with code 1."))
        MPI.Finalize()
        exit(code=1)
    end

    i, j, k = index = rank2index(my_rank, Rx, Ry, Rz)
    @debug "Rank: $my_rank, index: $index"

    #####
    ##### Construct local grid
    #####

    nx, ny, nz = Nx÷Rx, Ny÷Ry, Nz÷Rz
    lx, ly, lz = Lx/Rx, Ly/Ry, Lz/Rz

    x₁, x₂ = xL + (i-1)*lx, xL + i*lx
    y₁, y₂ = yL + (j-1)*ly, yL + j*ly
    z₁, z₂ = zL + (k-1)*lz, zL + k*lz

    @debug "Constructing local grid: n=($nx, $ny, $nz), x ∈ [$x₁, $x₂], y ∈ [$y₁, $y₂], z ∈ [$z₁, $z₂]"
    grid = RegularCartesianGrid(size=(nx, ny, nz), x=(x₁, x₂), y=(y₁, y₂), z=(z₁, z₂))

    #####
    ##### Construct local connectivity
    #####

    my_connectivity = construct_connectivity(index, ranks, boundary_conditions.u)
    @debug "Local connectivity: $my_connectivity"

    #####
    ##### Change appropriate boundary conditions to halo communication BCs
    #####

    @debug "Injecting halo communication boundary conditions..."
    boundary_conditions_with_communication = inject_halo_communication_boundary_conditions(boundary_conditions, my_rank, my_connectivity)

    #####
    ##### Construct local model
    #####

    my_model = Model(; grid = grid, boundary_conditions = boundary_conditions_with_communication,
                     model_kwargs...)

    return DistributedModel(index, ranks, my_model, my_connectivity)
end
