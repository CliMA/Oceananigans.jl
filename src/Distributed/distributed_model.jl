import MPI

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Grids

using Oceananigans.Grids: validate_tupled_argument
using Oceananigans.BoundaryConditions: BCType

import Oceananigans.BoundaryConditions:
    bctype_str, print_condition,
    fill_west_halo!, fill_east_halo!, fill_south_halo!,
    fill_north_halo!, fill_bottom_halo!, fill_top_halo!

#####
##### Architecture stuff
#####

# TODO: Put connectivity inside architecture? MPI should be initialize so you can construct it in there.
#       Might have to make it MultiCPU(; grid, ranks)

struct MultiCPU{R} <: AbstractArchitecture
    ranks :: R
end

MultiCPU(; ranks) = MultiCPU(ranks)

child_architecture(::MultiCPU) = CPU()

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
##### Rank connectivity graph
#####

struct RankConnectivity{E, W, N, S, T, B}
      east :: E
      west :: W
     north :: N
     south :: S
       top :: T
    bottom :: B
end

RankConnectivity(; east, west, north, south, top, bottom) =
    RankConnectivity(east, west, north, south, top, bottom)

function increment_index(i, R, topo)
    R == 1 && return nothing
    if i+1 > R
        if topo == Periodic
            return 1
        else
            return nothing
        end
    else
        return i+1
    end
end

function decrement_index(i, R, topo)
    R == 1 && return nothing
    if i-1 < 1
        if topo == Periodic
            return R
        else
            return nothing
        end
    else
        return i-1
    end
end

function RankConnectivity(model_index, ranks, topology)
    i, j, k = model_index
    Rx, Ry, Rz = ranks
    TX, TY, TZ = topology

    i_east  = increment_index(i, Rx, TX)
    i_west  = decrement_index(i, Rx, TX)
    j_north = increment_index(j, Ry, TY)
    j_south = decrement_index(j, Ry, TY)
    k_top   = increment_index(k, Rz, TZ)
    k_bot   = decrement_index(k, Rz, TZ)

    r_east  = isnothing(i_east)  ? nothing : index2rank(i_east, j, k, Rx, Ry, Rz)
    r_west  = isnothing(i_west)  ? nothing : index2rank(i_west, j, k, Rx, Ry, Rz)
    r_north = isnothing(j_north) ? nothing : index2rank(i, j_north, k, Rx, Ry, Rz)
    r_south = isnothing(j_south) ? nothing : index2rank(i, j_south, k, Rx, Ry, Rz)
    r_top   = isnothing(k_top)   ? nothing : index2rank(i, j, k_top, Rx, Ry, Rz)
    r_bot   = isnothing(k_bot)   ? nothing : index2rank(i, j, k_bot, Rx, Ry, Rz)

    return RankConnectivity(east=r_east, west=r_west, north=r_north,
                            south=r_south, top=r_top, bottom=r_bot)
end

#####
##### Halo communication boundary condition
#####

struct HaloCommunication <: BCType end

# const HaloCommunicationBC = BoundaryCondition{<:HaloCommunication}

bctype_str(::HaloCommunicationBC) ="HaloCommunication"

HaloCommunicationBoundaryCondition(val; kwargs...) = BoundaryCondition(HaloCommunication, val; kwargs...)

struct HaloCommunicationRanks{F, T}
    from :: F
      to :: T
end

HaloCommunicationRanks(; from, to) = HaloCommunicationRanks(from, to)

print_condition(hcr::HaloCommunicationRanks) = "(from rank $(hcr.from), to rank $(hcr.to))"

function inject_halo_communication_boundary_conditions(field_bcs, my_rank, connectivity)
    rank_east = connectivity.east
    rank_west = connectivity.west
    rank_north = connectivity.north
    rank_south = connectivity.south
    rank_top = connectivity.top
    rank_bottom = connectivity.bottom

    east_comm_ranks = HaloCommunicationRanks(from=my_rank, to=rank_east)
    west_comm_ranks = HaloCommunicationRanks(from=my_rank, to=rank_west)
    north_comm_ranks = HaloCommunicationRanks(from=my_rank, to=rank_north)
    south_comm_ranks = HaloCommunicationRanks(from=my_rank, to=rank_south)
    top_comm_ranks = HaloCommunicationRanks(from=my_rank, to=rank_top)
    bottom_comm_ranks = HaloCommunicationRanks(from=my_rank, to=rank_bottom)

    east_comm_bc = HaloCommunicationBoundaryCondition(east_comm_ranks)
    west_comm_bc = HaloCommunicationBoundaryCondition(west_comm_ranks)
    north_comm_bc = HaloCommunicationBoundaryCondition(north_comm_ranks)
    south_comm_bc = HaloCommunicationBoundaryCondition(south_comm_ranks)
    top_comm_bc = HaloCommunicationBoundaryCondition(top_comm_ranks)
    bottom_comm_bc = HaloCommunicationBoundaryCondition(bottom_comm_ranks)

    x_bcs = CoordinateBoundaryConditions(isnothing(rank_west) ? field_bcs.west : west_comm_bc,
                                         isnothing(rank_east) ? field_bcs.east : east_comm_bc)

    y_bcs = CoordinateBoundaryConditions(isnothing(rank_south) ? field_bcs.south : south_comm_bc,
                                         isnothing(rank_north) ? field_bcs.north : north_comm_bc)

    z_bcs = CoordinateBoundaryConditions(isnothing(rank_bottom) ? field_bcs.bottom : bottom_comm_bc,
                                         isnothing(rank_top) ? field_bcs.top : top_comm_bc)

    return FieldBoundaryConditions(x_bcs, y_bcs, z_bcs)
end

#####
##### Filling halos for halo communication boundary conditions
#####

# sides  = (:west, :east, :south, :north, :top, :bottom)
# coords = (:x,    :x,    :y,     :y,     :z,   :z)

# # Unfortunately can't call MPI.Comm_size(MPI.COMM_WORLD) before MPI.Init().
# const MAX_RANKS = 10^3

# # Define functions that return unique send and recv MPI tags for each side.
# for (i, side) in enumerate(sides)
#     send_tag_fn_name = Symbol(side, :_halo_comm_bc_send_tag)
#     recv_tag_fn_name = Symbol(side, :_halo_comm_bc_recv_tag)
#     @eval begin
#         @inline $send_tag_fn_name(bc) = 6 * (MAX_RANKS * bc.condition.rank_from + bc.condition.rank_to)   + $i
#         @inline $recv_tag_fn_name(bc) = 6 * (MAX_RANKS * bc.condition.rank_to   + bc.condition.rank_from) + $i
#     end
# end

# @inline   west_send_buffer(c, N, H) = c.parent[N+1:N+H, :, :]
# @inline   east_send_buffer(c, N, H) = c.parent[1+H:2H,  :, :]
# @inline  south_send_buffer(c, N, H) = c.parent[:, N+1:N+H, :]
# @inline  north_send_buffer(c, N, H) = c.parent[:, 1+H:2H,  :]
# @inline    top_send_buffer(c, N, H) = c.parent[:, :,  1+H:2H]
# @inline bottom_send_buffer(c, N, H) = c.parent[:, :, N+1:N+H]

# @inline west_recv_buffer(grid)  = zeros(grid.Hx, grid.Ty, grid.Tz)
# @inline south_recv_buffer(grid) = zeros(grid.Tx, grid.Hy, grid.Tz)
# @inline top_recv_buffer(grid)   = zeros(grid.Tx, grid.Ty, grid.Hz)

# const   east_recv_buffer =  west_recv_buffer
# const  north_recv_buffer = south_recv_buffer
# const bottom_recv_buffer =   top_recv_buffer

# @inline   copy_recv_buffer_into_west_halo!(c, N, H, buf) = (c.parent[    1:H,    :, :] .= buf)
# @inline   copy_recv_buffer_into_east_halo!(c, N, H, buf) = (c.parent[N+H+1:N+2H, :, :] .= buf)
# @inline  copy_recv_buffer_into_south_halo!(c, N, H, buf) = (c.parent[:,     1:H,    :] .= buf)
# @inline  copy_recv_buffer_into_north_halo!(c, N, H, buf) = (c.parent[:, N+H+1:N+2H, :] .= buf)
# @inline copy_recv_buffer_into_bottom_halo!(c, N, H, buf) = (c.parent[:, :,     1:H   ] .= buf)
# @inline    copy_recv_buffer_into_top_halo!(c, N, H, buf) = (c.parent[:, :, N+H+1:N+2H] .= buf)

# for (x, side) in zip(coords, sides)
#     H = Symbol(:H, x)
#     N = Symbol(:N, x)

#     fill_fn_name     = Symbol(:fill_, side, :_halo!)
#     send_buf_fn_name = Symbol(side, :_send_buffer)
#     recv_buf_fn_name = Symbol(side, :_recv_buffer)
#     send_tag_fn_name = Symbol(side, :_halo_comm_bc_send_tag)
#     recv_tag_fn_name = Symbol(side, :_halo_comm_bc_recv_tag)
#     copy_buf_fn_name = Symbol(:copy_recv_buffer_into_, side, :_halo!)

#     @eval begin
#         function $fill_fn_name(c, bc::HaloCommunicationBC, arch, grid, args...)
#             send_buffer = $send_buf_fn_name(c, grid.$(N), grid.$(H))
#             recv_buffer = $recv_buf_fn_name(grid)

#             send_tag = $send_tag_fn_name(bc)
#             recv_tag = $recv_tag_fn_name(bc)

#             my_rank = bc.condition.rank_from
#             rank_send_to = rank_recv_from = bc.condition.rank_to

#             @info "MPI.Isend: my_rank=$my_rank, rank_send_to=$rank_send_to, send_tag=$send_tag"
#             MPI.Isend(send_buffer, rank_send_to, send_tag, MPI.COMM_WORLD)
#             @info "MPI.Isend: done!"

#             @info "MPI.Recv!: my_rank=$my_rank, rank_recv_from=$rank_recv_from, recv_tag=$recv_tag"
#             MPI.Recv!(recv_buffer, rank_recv_from, recv_tag, MPI.COMM_WORLD)
#             @info "MPI.Recv! done!"

#             # @info "Sendrecv!: my_rank=$my_rank, rank_send_to=rank_recv_from=$rank_send_to, " *
#             #       "send_tag=$send_tag, recv_tag=$recv_tag"
#             #
#             # MPI.Sendrecv!(send_buffer, rank_send_to,   send_tag,
#             #               recv_buffer, rank_recv_from, recv_tag,
#             #               MPI.COMM_WORLD)
#             #
#             # @info "Sendrecv!: my_rank=$my_rank done!"

#             $copy_buf_fn_name(c, grid.$(N), grid.$(H), recv_buffer)
#         end
#     end
# end

#####
##### Distributed model struct and constructor
#####

struct DistributedModel{A, I, M, R, G}
    architecture :: A
           index :: I
           ranks :: R
           model :: M
    connectivity :: G
end

function DistributedModel(; architecture, grid, boundary_conditions=nothing, model_kwargs...)
    ranks = architecture.ranks

    validate_tupled_argument(ranks, Int, "ranks")

    Nx, Ny, Nz = size(grid)

    # Pull out left and right endpoints for full model.
    xL, xR = grid.xF[1], grid.xF[Nx+1]
    yL, yR = grid.yF[1], grid.yF[Ny+1]
    zL, zR = grid.zF[1], grid.zF[Nz+1]
    Lx, Ly, Lz = length(grid)

    Rx, Ry, Rz = ranks
    total_ranks = Rx*Ry*Rz

    comm = MPI.COMM_WORLD

    mpi_ranks = MPI.Comm_size(comm)
    my_rank   = MPI.Comm_rank(comm)

    if total_ranks != mpi_ranks
        throw(ArgumentError("ranks=($Rx, $Ry, $Rz) [$total_ranks total] inconsistent " *
                            "with number of MPI ranks: $mpi_ranks. Exiting with return code 1."))
        MPI.Finalize()
        exit(code=1)
    end

    i, j, k = index = rank2index(my_rank, Rx, Ry, Rz)
    @info "My rank: $my_rank, my index: $index"

    #####
    ##### Construct local grid
    #####

    # Make sure we can put an integer number of grid points in each rank.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)

    nx, ny, nz = Nx÷Rx, Ny÷Ry, Nz÷Rz
    lx, ly, lz = Lx/Rx, Ly/Ry, Lz/Rz

    x₁, x₂ = xL + (i-1)*lx, xL + i*lx
    y₁, y₂ = yL + (j-1)*ly, yL + j*ly
    z₁, z₂ = zL + (k-1)*lz, zL + k*lz

    @info "Constructing local grid: n=($nx, $ny, $nz), x ∈ [$x₁, $x₂], y ∈ [$y₁, $y₂], z ∈ [$z₁, $z₂]"
    my_grid = RegularCartesianGrid(topology=topology(grid), size=(nx, ny, nz), x=(x₁, x₂), y=(y₁, y₂), z=(z₁, z₂))

    #####
    ##### Construct local connectivity
    #####

    my_connectivity = RankConnectivity(index, ranks, topology(grid))
    @info "Local connectivity: $my_connectivity"

    #####
    ##### Change appropriate boundary conditions to halo communication BCs
    #####

    # FIXME: Stop assuming (u, v, w, T, S).

    bcs = isnothing(boundary_conditions) ? NamedTuple() : boundary_conditions

    bcs = (
        u = haskey(bcs, :u) ? bcs.u : UVelocityBoundaryConditions(grid),
        v = haskey(bcs, :v) ? bcs.v : VVelocityBoundaryConditions(grid),
        w = haskey(bcs, :w) ? bcs.w : WVelocityBoundaryConditions(grid),
        T = haskey(bcs, :T) ? bcs.T : TracerBoundaryConditions(grid),
        S = haskey(bcs, :S) ? bcs.S : TracerBoundaryConditions(grid)
    )

    @debug "Injecting halo communication boundary conditions..."

    communicative_bcs = (
        u = inject_halo_communication_boundary_conditions(bcs.u, my_rank, my_connectivity),
        v = inject_halo_communication_boundary_conditions(bcs.v, my_rank, my_connectivity),
        w = inject_halo_communication_boundary_conditions(bcs.w, my_rank, my_connectivity),
        T = inject_halo_communication_boundary_conditions(bcs.T, my_rank, my_connectivity),
        S = inject_halo_communication_boundary_conditions(bcs.S, my_rank, my_connectivity)
    )

    #####
    ##### Construct local model
    #####

    my_model = IncompressibleModel(;
               architecture = child_architecture(architecture),
                       grid = my_grid,
        boundary_conditions = communicative_bcs,
        model_kwargs...
    )

    return DistributedModel(architecture, index, ranks, my_model, my_connectivity)
end

function Base.show(io::IO, dm::DistributedModel)
    print(io, "DistributedModel with $(dm.ranks) ranks")
end
