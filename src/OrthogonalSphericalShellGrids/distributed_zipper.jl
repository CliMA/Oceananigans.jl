using MPI
using Oceananigans.BoundaryConditions: get_boundary_kernels, DistributedCommunication, North, SouthAndNorth
using Oceananigans.DistributedComputations: cooperative_waitall!, distributed_fill_halo_event!
using Oceananigans.DistributedComputations: CommunicationBuffers, fill_corners!, loc_id
using Oceananigans.DistributedComputations: ranks, index2rank,
    OneDBuffer, TwoDBuffer, CornerBuffer,
    y_communication_buffer, x_communication_buffer, corner_communication_buffer,
    _fill_north_send_buffer!, _fill_south_send_buffer!,
    _fill_southwest_send_buffer!, _fill_southeast_send_buffer!,
    _fill_northwest_send_buffer!, _fill_northeast_send_buffer!,
    _recv_from_west_buffer!, _recv_from_east_buffer!,
    _recv_from_south_buffer!, _recv_from_north_buffer!,
    _recv_from_southwest_buffer!, _recv_from_southeast_buffer!,
    _recv_from_northwest_buffer!, _recv_from_northeast_buffer!
using Oceananigans.Fields: instantiated_location

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication!,
    CommunicationBuffers, fill_send_buffers!, recv_from_buffers!

@inline instantiate(T::DataType) = T()
@inline instantiate(T) = T

const DistributedZipper = BoundaryCondition{<:DistributedCommunication, <:ZipperHaloCommunicationRanks}

@inline is_north_fold_rank(grid) = architecture(grid).local_index[2] == ranks(architecture(grid))[2]

#####
##### Extended north buffer allocation for fold-exact matching
#####
##### Top y-ranks get Hy+2 north buffers to carry enough source rows for ALL 8 serial
##### fold cases (UPivot + FPivot × Face/Center × x/y). MPI pairing is same-sided
##### (north.send ↔ fold_partner.north.recv), so both fold partners get Hy+2.
#####

function CommunicationBuffers(grid::DistributedTripolarGridOfSomeKind, data,
                              boundary_conditions::FieldBoundaryConditions)
    Hx, Hy, Hz = halo_size(grid)
    arch = architecture(grid)

    Hy_north = is_north_fold_rank(grid) ? Hy + 2 : Hy

    west  = x_communication_buffer(arch, grid, data, Hx, boundary_conditions.west)
    east  = x_communication_buffer(arch, grid, data, Hx, boundary_conditions.east)
    south = y_communication_buffer(arch, grid, data, Hy, boundary_conditions.south)
    north = y_communication_buffer(arch, grid, data, Hy_north, boundary_conditions.north)

    sw = corner_communication_buffer(arch, grid, data, Hx, Hy, west, south)
    se = corner_communication_buffer(arch, grid, data, Hx, Hy, east, south)
    nw = corner_communication_buffer(arch, grid, data, Hx, Hy_north, west, north)
    ne = corner_communication_buffer(arch, grid, data, Hx, Hy_north, east, north)

    return CommunicationBuffers(west, east, south, north, sw, se, nw, ne)
end

#####
##### Extended north send: Hy+2 rows = parent[Ny-1:Ny+Hy]
#####

_fill_north_send_extended!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, :, Ny-1:Ny+Hy, :)
_fill_north_send_extended!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, 1+Hx:Nx+Hx, Ny-1:Ny+Hy, :)
_fill_north_send_extended!(c, ::Nothing, Hx, Hy, Nx, Ny) = nothing

_fill_nw_send_extended!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, 1+Hx:2Hx, Ny-1:Ny+Hy, :)
_fill_ne_send_extended!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, 1+Nx:Nx+Hx, Ny-1:Ny+Hy, :)
_fill_nw_send_extended!(c, ::Nothing, Hx, Hy, Nx, Ny) = nothing
_fill_ne_send_extended!(c, ::Nothing, Hx, Hy, Nx, Ny) = nothing

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers,
                            grid::DistributedTripolarGridOfSomeKind, ::North)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    if is_north_fold_rank(grid)
        _fill_north_send_extended!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    else
        _fill_north_send_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    end
end

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers,
                            grid::DistributedTripolarGridOfSomeKind, ::SouthAndNorth)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    _fill_south_send_buffer!(parent(c), buff.south, Hx, Hy, Nx, Ny)
    if is_north_fold_rank(grid)
        _fill_north_send_extended!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    else
        _fill_north_send_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    end
end

function fill_send_buffers!(c::OffsetArray, buff::CommunicationBuffers,
                            grid::DistributedTripolarGridOfSomeKind, ::Val{:corners})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    _fill_southwest_send_buffer!(parent(c), buff.southwest, Hx, Hy, Nx, Ny)
    _fill_southeast_send_buffer!(parent(c), buff.southeast, Hx, Hy, Nx, Ny)
    if is_north_fold_rank(grid)
        _fill_nw_send_extended!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
        _fill_ne_send_extended!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)
    else
        _fill_northwest_send_buffer!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
        _fill_northeast_send_buffer!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)
    end
    return nothing
end

#####
##### Recv overrides for top ranks: copy last Hy rows from Hy+2 buffer into parent.
##### This populates corner x-regions for the in-place reversal in switch_north_halos!,
##### while interior x will be overwritten from the full buffer by switch_north_halos!.
#####

# Fold-aware recv: copy last Hy rows from extended Hy+2 buffer to standard Hy parent halo
_recv_from_north_buffer_fold!(c, buff::OneDBuffer, Hx, Hy, Nx, Ny) =
    view(c, :, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, size(buff.recv, 2)-Hy+1:size(buff.recv, 2), :)
_recv_from_north_buffer_fold!(c, buff::TwoDBuffer, Hx, Hy, Nx, Ny) =
    view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, size(buff.recv, 2)-Hy+1:size(buff.recv, 2), :)
_recv_from_north_buffer_fold!(c, ::Nothing, Hx, Hy, Nx, Ny) = nothing

_recv_from_northwest_buffer_fold!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) =
    view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, size(buff.recv, 2)-Hy+1:size(buff.recv, 2), :)
_recv_from_northeast_buffer_fold!(c, buff::CornerBuffer, Hx, Hy, Nx, Ny) =
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, size(buff.recv, 2)-Hy+1:size(buff.recv, 2), :)
_recv_from_northwest_buffer_fold!(c, ::Nothing, Hx, Hy, Nx, Ny) = nothing
_recv_from_northeast_buffer_fold!(c, ::Nothing, Hx, Hy, Nx, Ny) = nothing

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers,
                            grid::DistributedTripolarGridOfSomeKind, ::North)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    if is_north_fold_rank(grid)
        _recv_from_north_buffer_fold!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    else
        _recv_from_north_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    end
    return nothing
end

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers,
                            grid::DistributedTripolarGridOfSomeKind, ::SouthAndNorth)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    _recv_from_south_buffer!(parent(c), buff.south, Hx, Hy, Nx, Ny)
    if is_north_fold_rank(grid)
        _recv_from_north_buffer_fold!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    else
        _recv_from_north_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    end
    return nothing
end

function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers,
                            grid::DistributedTripolarGridOfSomeKind, ::Val{:corners})
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    _recv_from_southwest_buffer!(parent(c), buff.southwest, Hx, Hy, Nx, Ny)
    _recv_from_southeast_buffer!(parent(c), buff.southeast, Hx, Hy, Nx, Ny)
    if is_north_fold_rank(grid)
        _recv_from_northwest_buffer_fold!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
        _recv_from_northeast_buffer_fold!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)
    else
        _recv_from_northwest_buffer!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
        _recv_from_northeast_buffer!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)
    end
    return nothing
end

# No-side variant used by synchronize_communication!
function recv_from_buffers!(c::OffsetArray, buff::CommunicationBuffers,
                            grid::DistributedTripolarGridOfSomeKind)
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = size(grid)
    _recv_from_west_buffer!(parent(c), buff.west, Hx, Hy, Nx, Ny)
    _recv_from_east_buffer!(parent(c), buff.east, Hx, Hy, Nx, Ny)
    _recv_from_south_buffer!(parent(c), buff.south, Hx, Hy, Nx, Ny)
    if is_north_fold_rank(grid)
        _recv_from_north_buffer_fold!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    else
        _recv_from_north_buffer!(parent(c), buff.north, Hx, Hy, Nx, Ny)
    end
    _recv_from_southwest_buffer!(parent(c), buff.southwest, Hx, Hy, Nx, Ny)
    _recv_from_southeast_buffer!(parent(c), buff.southeast, Hx, Hy, Nx, Ny)
    if is_north_fold_rank(grid)
        _recv_from_northwest_buffer_fold!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
        _recv_from_northeast_buffer_fold!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)
    else
        _recv_from_northwest_buffer!(parent(c), buff.northwest, Hx, Hy, Nx, Ny)
        _recv_from_northeast_buffer!(parent(c), buff.northeast, Hx, Hy, Nx, Ny)
    end
    return nothing
end

#####
##### Distributed fold (zipper) halo fill
#####
# The fold (zipper) at the north boundary of a TripolarGrid rotates rows above
# the fold line by 180° around the pivot point, filling them with x-reversed
# data from source rows below:
#
#   Ny+Hy ─▶  ┌───────────────────────┐  ← fold row Hy (dest, filled from buffer)
#             │  ...                  │
#   Ny+1  ─▶  ├───────────────────────┤  ← fold row 1  (dest, filled from buffer)
#   Ny    ─▶  ╞═══════════════════════╡  ← fold line    (half-row substitution)
#   Ny-1  ─▶  ├───────────────────────┤  ← source row 1 (read by partner buffer)
#             │  ...                  │
#
# In serial, the fold kernel operates element-wise over the full array, then
# the Periodic x-halo exchange fills fold-row x-halos. In distributed:
#
#   1. Async MPI exchanges all halos (x, y, corners) simultaneously
#   2. switch_north_halos! runs the fold → overwrites fold halo rows
#   3. Fold-row x-halos are now stale (still hold pre-fold values)
#
# To fix this, switch_north_halos! runs four steps:
#   fill_north_fold_halo!              — fold halo rows from partner buffer
#   fill_half_north_fold_line!         — fold-line half-row substitution
#   fill_north_fold_halo_west_column!  — FC/FF column 1 fix (MPI with conjugate rank)
#   exchange_north_fold_halos!         — re-exchange x-halos at fold rows (MPI with x-neighbors)
#####

switch_north_halos!(c, north_bc, grid, loc, buffers) = nothing

function switch_north_halos!(c, north_bc::DistributedZipper, grid, loc, buffers)
    sign = north_bc.condition.sign
    fold_topo = grid.conformal_mapping.fold_topology
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    buf_x = buffer_x_interior(buffers.north, Hx, Nx)

    # Step 1: Fill fold halo rows from partner buffer
    fill_north_fold_halo!(parent(c), buffers.north.recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x)

    # Step 2: Fold-line half-row substitution (CC/FC at UPivot, CF/FF at FPivot)
    # Serial kernels overwrite the right half (global i > Nx_global/2) of the fold line
    # with x-reversed values from the left half. In distributed:
    # - Multiple x-ranks: left rank keeps values, right rank replaces all from buffer.
    # - Single x-rank (Rx==1): only replace the right half (i > Nx÷2) to match serial.
    arch = architecture(grid)
    Rx = ranks(arch)[1]
    if Rx == 1
        fill_half_north_fold_line!(parent(c), buffers.north.recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x, Nx ÷ 2 + 1)
    elseif arch.local_index[1] > Rx ÷ 2
        fill_half_north_fold_line!(parent(c), buffers.north.recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x, 1)
    end

    # Step 3: FC/FF column 1 fix (no-op for CC/CF)
    fill_north_fold_halo_west_column!(parent(c), loc, sign, grid, Nx, Ny, Hx, Hy, fold_topo)

    # Step 4: Re-exchange x-halos at fold rows with east/west x-neighbors
    exchange_north_fold_halos!(parent(c), loc, grid, Nx, Ny, Hx, Hy, fold_topo)

    return nothing
end

# Buffer x-range dispatch (OneDBuffer has full x, TwoDBuffer has interior only)
@inline buffer_x_interior(::OneDBuffer, Hx, Nx) = Hx+1:Nx+Hx
@inline buffer_x_interior(::TwoDBuffer, Hx, Nx) = 1:Nx

# Fold parameters: (nj, dest_offset, src_k_fn) for each (fold_topo, y_location) pair.
# buffer[:, k, :] where k=1..Hy+2: k=1 = partner offset ny-Hy-1, k=Hy+2 = partner offset ny.
# Partner offset ny-m → buffer k = Hy+2-m.
#
# Pattern A (UPivot, YFace): nj=Hy, dest(j)=Ny+Hy+j, src_k(j)=Hy+3-j
# Pattern B (UPivot YCenter, FPivot YFace): nj=Hy, dest(j)=Ny+Hy+j, src_k(j)=Hy+2-j
# Pattern C (FPivot, YCenter): nj=Hy+1, dest(j)=Ny+Hy-1+j, src_k(j)=Hy+2-j

# Pattern A: UPivot, YFace (source: partner ny-j+1, dest: Ny+j)
@inline fold_nj(     ::Tuple{<:Any, <:Face,   <:Any}, Hy, ::Type{RightCenterFolded}) = Hy
@inline fold_dest_y( ::Tuple{<:Any, <:Face,   <:Any}, Ny, Hy, j, ::Type{RightCenterFolded}) = Ny + Hy + j
@inline fold_src_k(  ::Tuple{<:Any, <:Face,   <:Any}, Hy, j, ::Type{RightCenterFolded}) = Hy + 3 - j

# Pattern B: UPivot, YCenter (source: partner ny-j, dest: Ny+j)
@inline fold_nj(     ::Tuple{<:Any, <:Center, <:Any}, Hy, ::Type{RightCenterFolded}) = Hy
@inline fold_dest_y( ::Tuple{<:Any, <:Center, <:Any}, Ny, Hy, j, ::Type{RightCenterFolded}) = Ny + Hy + j
@inline fold_src_k(  ::Tuple{<:Any, <:Center, <:Any}, Hy, j, ::Type{RightCenterFolded}) = Hy + 2 - j

# Pattern B: FPivot, YFace (same indexing as UPivot YCenter)
@inline fold_nj(     ::Tuple{<:Any, <:Face,   <:Any}, Hy, ::Type{RightFaceFolded}) = Hy
@inline fold_dest_y( ::Tuple{<:Any, <:Face,   <:Any}, Ny, Hy, j, ::Type{RightFaceFolded}) = Ny + Hy + j
@inline fold_src_k(  ::Tuple{<:Any, <:Face,   <:Any}, Hy, j, ::Type{RightFaceFolded}) = Hy + 2 - j

# Pattern C: FPivot, YCenter (source: partner ny-j, dest: Ny-1+j, Hy+1 rows)
@inline fold_nj(     ::Tuple{<:Any, <:Center, <:Any}, Hy, ::Type{RightFaceFolded}) = Hy + 1
@inline fold_dest_y( ::Tuple{<:Any, <:Center, <:Any}, Ny, Hy, j, ::Type{RightFaceFolded}) = Ny + Hy - 1 + j
@inline fold_src_k(  ::Tuple{<:Any, <:Center, <:Any}, Hy, j, ::Type{RightFaceFolded}) = Hy + 2 - j

# CC/CF fold: all Nx columns from reversed partner buffer (i' = Nx+1-i).
@inline function fill_north_fold_halo_row!(c, sign, Hx, Nx, dest_y, north_recv, buf_x, src_k,
                                           ::Tuple{<:Center, <:Any, <:Any})
    view(c, Hx+1:Hx+Nx, dest_y:dest_y, :) .= sign .* reverse(view(north_recv, buf_x, src_k:src_k, :), dims=1)
end

# FC/FF fold: columns 2..Nx from reversed partner buffer (i' = Nx+2-i).
# Column 1 handled by fill_north_fold_halo_west_column! because the FC/FF fixed point
# (global i=1 → i=1) only holds for x-rank 0; other x-ranks need column 1
# from a different rank (the "conjugate" — see fill_north_fold_halo_west_column!).
@inline function fill_north_fold_halo_row!(c, sign, Hx, Nx, dest_y, north_recv, buf_x, src_k,
                                           ::Tuple{<:Face, <:Any, <:Any})
    view(c, Hx+2:Hx+Nx, dest_y:dest_y, :) .= sign .* reverse(view(north_recv, buf_x[2]:buf_x[end], src_k:src_k, :), dims=1)
end

# Fold-line half-row substitution from buffer.
# Called only on right-half x-ranks (or the single x-rank when Rx==1 with fold_i_start > 1).
# Replaces fold-line values with x-reversed partner data from buffer.
# Buffer k=Hy+2 holds the partner's fold-line row (partner parent y = Ny+Hy).
fill_half_north_fold_line!(c, recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x, fold_i_start) = nothing

# CC/CF: all Nx columns from reversed partner buffer
@inline function _fill_fold_line_all_columns!(c, recv, sign, Nx, Hx, fold_y, fold_k, buf_x, fold_i_start)
    reversed = sign .* reverse(view(recv, buf_x, fold_k:fold_k, :), dims=1)
    view(c, Hx+fold_i_start:Hx+Nx, fold_y:fold_y, :) .= view(reversed, fold_i_start:Nx, :, :)
end

# FC/FF: columns 2..Nx from reversed partner buffer (column 1 handled by fill_north_fold_halo_west_column!)
@inline function _fill_fold_line_skip_column1!(c, recv, sign, Nx, Hx, fold_y, fold_k, buf_x, fold_i_start)
    i_start = max(2, fold_i_start)
    reversed = sign .* reverse(view(recv, buf_x[2]:buf_x[end], fold_k:fold_k, :), dims=1)
    view(c, Hx+i_start:Hx+Nx, fold_y:fold_y, :) .= view(reversed, i_start-1:Nx-1, :, :)
end

# CF (Center, Face), FPivot
fill_half_north_fold_line!(c, recv, ::Tuple{<:Center, <:Face, <:Any}, sign, Nx, Ny, Hx, Hy,
                            ::Type{RightFaceFolded}, buf_x, fi) =
    _fill_fold_line_all_columns!(c, recv, sign, Nx, Hx, Ny+Hy, Hy+2, buf_x, fi)

# CC (Center, Center), UPivot
fill_half_north_fold_line!(c, recv, ::Tuple{<:Center, <:Center, <:Any}, sign, Nx, Ny, Hx, Hy,
                            ::Type{RightCenterFolded}, buf_x, fi) =
    _fill_fold_line_all_columns!(c, recv, sign, Nx, Hx, Ny+Hy, Hy+2, buf_x, fi)

# FF (Face, Face), FPivot
fill_half_north_fold_line!(c, recv, ::Tuple{<:Face, <:Face, <:Any}, sign, Nx, Ny, Hx, Hy,
                            ::Type{RightFaceFolded}, buf_x, fi) =
    _fill_fold_line_skip_column1!(c, recv, sign, Nx, Hx, Ny+Hy, Hy+2, buf_x, fi)

# FC (Face, Center), UPivot
fill_half_north_fold_line!(c, recv, ::Tuple{<:Face, <:Center, <:Any}, sign, Nx, Ny, Hx, Hy,
                            ::Type{RightCenterFolded}, buf_x, fi) =
    _fill_fold_line_skip_column1!(c, recv, sign, Nx, Hx, Ny+Hy, Hy+2, buf_x, fi)

# Fold-line parent y for cases that need fold-line overwrite; nothing otherwise.
_fold_line_parent_y(loc, Ny, Hy, fold_topo) = nothing
_fold_line_parent_y(::Tuple{<:Any, <:Center, <:Any}, Ny, Hy, ::Type{RightCenterFolded}) = Ny + Hy
_fold_line_parent_y(::Tuple{<:Any, <:Face, <:Any}, Ny, Hy, ::Type{RightFaceFolded}) = Ny + Hy

# Contiguous parent y-range of all fold-affected rows (halo rows + fold line).
# Used by exchange_north_fold_halos! to batch MPI exchanges.
@inline function fold_y_range(loc, Ny, Hy, fold_topo)
    nj = fold_nj(loc, Hy, fold_topo)
    y_first = fold_dest_y(loc, Ny, Hy, 1, fold_topo)
    y_last  = fold_dest_y(loc, Ny, Hy, nj, fold_topo)
    fold_line_y = _fold_line_parent_y(loc, Ny, Hy, fold_topo)
    y_start = fold_line_y !== nothing ? min(y_first, fold_line_y) : y_first
    return y_start:y_last
end

# Step 1: Fill fold halo rows (above the fold line) from partner buffer.
# Writes non-halo columns of each fold halo row with x-reversed partner data.
# CC/CF: all Nx columns; FC/FF: columns 2..Nx (column 1 handled separately).
function fill_north_fold_halo!(c, north_recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x)
    nj = fold_nj(loc, Hy, fold_topo)

    for j in 1:nj
        dest_y = fold_dest_y(loc, Ny, Hy, j, fold_topo)
        src_k  = fold_src_k(loc, Hy, j, fold_topo)
        fill_north_fold_halo_row!(c, sign, Hx, Nx, dest_y, north_recv, buf_x, src_k, loc)
    end

    return nothing
end

#####
##### Step 3: FC/FF column 1 fix — fill_north_fold_halo_west_column!
#####
# The FC/FF fold maps global i → Nx_global + 2 - i (shifted by 1 from CC/CF).
# Column 1 of each x-rank maps to column 1 of a "conjugate" x-rank:
#   rx_conj = (Rx - rx) % Rx   (0-based x-rank indices)
# Fixed points (self-map): rx=0, and rx=Rx/2 (when Rx is even).
# For non-fixed points, column 1 data must be exchanged via MPI with the conjugate.
# No-op for CC/CF (Center first component) since their fold has no column 1 shift.

# CC/CF: no column 1 fix needed
fill_north_fold_halo_west_column!(c, loc, sign, grid, Nx, Ny, Hx, Hy, fold_topo) = nothing
fill_north_fold_halo_west_column!(c, ::Tuple{<:Center, <:Any, <:Any}, sign, grid, Nx, Ny, Hx, Hy, fold_topo) = nothing

# FC/FF: fill column 1 at fold halo rows via MPI with conjugate x-rank.
# The fold line column 1 is only WRITTEN on right-half ranks (same condition as step 2),
# but both conjugate ranks must PARTICIPATE in the MPI exchange to avoid deadlock.
function fill_north_fold_halo_west_column!(c, loc::Tuple{<:Face, <:Any, <:Any}, sign, grid, Nx, Ny, Hx, Hy, fold_topo)
    arch = architecture(grid)
    Rx, Ry, Rz = ranks(arch)
    rx = arch.local_index[1] - 1  # 0-based x-rank
    rx_conj = mod(Rx - rx, Rx)    # 0-based conjugate x-rank
    nj = fold_nj(loc, Hy, fold_topo)
    col1 = Hx + 1  # parent index of column 1

    fold_line_y = _fold_line_parent_y(loc, Ny, Hy, fold_topo)
    is_right_half = (Rx == 1) || (arch.local_index[1] > Rx ÷ 2)

    if rx_conj == rx
        # Fixed point: column 1 maps to self — local copy
        for j in 1:nj
            dest_y = fold_dest_y(loc, Ny, Hy, j, fold_topo)
            src_y  = Ny + fold_src_k(loc, Hy, j, fold_topo) - 2
            view(c, col1:col1, dest_y:dest_y, :) .= sign .* view(c, col1:col1, src_y:src_y, :)
        end
        if fold_line_y !== nothing && is_right_half
            view(c, col1:col1, fold_line_y:fold_line_y, :) .*= sign
        end
    else
        # Non-fixed point: exchange column 1 with conjugate rank
        ry, rz = arch.local_index[2], arch.local_index[3]
        conj_rank = index2rank(rx_conj + 1, ry, rz, Rx, Ry, Rz)
        comm = arch.communicator
        send_buf = similar(c, 1, 1, size(c, 3))
        recv_buf = similar(send_buf)

        tag = 10000
        for j in 1:nj
            dest_y = fold_dest_y(loc, Ny, Hy, j, fold_topo)
            src_y  = Ny + fold_src_k(loc, Hy, j, fold_topo) - 2
            send_buf .= view(c, col1:col1, src_y:src_y, :)
            MPI.Sendrecv!(send_buf, recv_buf, comm; dest=conj_rank, source=conj_rank, sendtag=tag-j, recvtag=tag-j)
            view(c, col1:col1, dest_y:dest_y, :) .= sign .* recv_buf
        end
        if fold_line_y !== nothing
            send_buf .= view(c, col1:col1, fold_line_y:fold_line_y, :)
            MPI.Sendrecv!(send_buf, recv_buf, comm; dest=conj_rank, source=conj_rank, sendtag=tag-nj-1, recvtag=tag-nj-1)
            if is_right_half
                view(c, col1:col1, fold_line_y:fold_line_y, :) .= sign .* recv_buf
            end
        end
    end

    return nothing
end

#####
##### Step 4: Re-exchange x-halos at fold rows — exchange_north_fold_halos!
#####
# After the fold writes (steps 1-3), the x-halos at fold rows are stale because
# the initial async MPI x-halo exchange ran BEFORE the fold overwrote these rows.
# This function performs a targeted x-halo exchange for just the fold-affected rows.

function exchange_north_fold_halos!(c, loc, grid, Nx, Ny, Hx, Hy, fold_topo)
    arch = architecture(grid)
    Rx = ranks(arch)[1]
    Rx == 1 && return nothing

    comm = arch.communicator
    rx = arch.local_index[1]
    west_rank = index2rank(mod1(rx - 1, Rx), arch.local_index[2], arch.local_index[3], ranks(arch)...)
    east_rank = index2rank(mod1(rx + 1, Rx), arch.local_index[2], arch.local_index[3], ranks(arch)...)

    yr = fold_y_range(loc, Ny, Hy, fold_topo)

    # Send westmost Hx columns to west neighbor, recv from east neighbor into east halo
    send_west = copy(view(c, Hx+1:2Hx, yr, :))
    recv_east = similar(send_west)
    MPI.Sendrecv!(send_west, recv_east, comm; dest=west_rank, source=east_rank, sendtag=9997, recvtag=9997)
    view(c, Nx+Hx+1:Nx+2Hx, yr, :) .= recv_east

    # Send eastmost Hx columns to east neighbor, recv from west neighbor into west halo
    send_east = copy(view(c, Nx+1:Nx+Hx, yr, :))
    recv_west = similar(send_east)
    MPI.Sendrecv!(send_east, recv_west, comm; dest=east_rank, source=west_rank, sendtag=9996, recvtag=9996)
    view(c, 1:Hx, yr, :) .= recv_west

    return nothing
end

#####
##### fill_halo_regions! and synchronize_communication! overrides
#####

# Disambiguation
fill_halo_regions!(c::OffsetArray, ::Nothing, indices, loc, ::DistributedTripolarGridOfSomeKind, args...; kwargs...) = nothing

function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::DistributedTripolarGridOfSomeKind, buffers::CommunicationBuffers, args...; kwargs...)
    arch = architecture(grid)
    kernels!, ordered_bcs = get_boundary_kernels(bcs, c, grid, loc, indices)

    number_of_tasks = length(kernels!)
    outstanding_requests = length(arch.mpi_requests)

    for task = 1:number_of_tasks
        @inbounds distributed_fill_halo_event!(c, kernels![task], ordered_bcs[task], loc, grid, buffers, args...; kwargs...)
    end

    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers, args...; kwargs...)

    # We increment the request counter only if we have actually initiated the MPI communication.
    # This is the case only if at least one of the boundary conditions is a distributed communication
    # boundary condition (DCBCT) _and_ the `only_local_halos` keyword argument is false.
    if length(arch.mpi_requests) > outstanding_requests
        arch.mpi_tag[] += 1
    end

    if arch.mpi_tag[] == 0 # The communication has been reset, switch the north halos!
        north_bc = bcs.north
        switch_north_halos!(c, north_bc, grid, loc, buffers)
    end

    return nothing
end

function synchronize_communication!(field::Field{<:Any, <:Any, <:Any, <:Any, <:DistributedTripolarGridOfSomeKind})
    arch = architecture(field.grid)

    # Wait for outstanding requests
    if !isempty(arch.mpi_requests)
        cooperative_waitall!(arch.mpi_requests)

        # Reset MPI tag
        arch.mpi_tag[] = 0

        # Reset MPI requests
        empty!(arch.mpi_requests)
    end

    recv_from_buffers!(field.data, field.communication_buffers, field.grid)

    north_bc = field.boundary_conditions.north
    switch_north_halos!(field.data, north_bc, field.grid, instantiated_location(field), field.communication_buffers)

    return nothing
end
