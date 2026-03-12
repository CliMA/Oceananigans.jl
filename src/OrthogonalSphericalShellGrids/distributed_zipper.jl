using Oceananigans.BoundaryConditions: get_boundary_kernels, DistributedCommunication, North, SouthAndNorth
using Oceananigans.DistributedComputations: cooperative_waitall!, distributed_fill_halo_event!
using Oceananigans.DistributedComputations: CommunicationBuffers, fill_corners!, loc_id
using Oceananigans.DistributedComputations: ranks,
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
##### switch_north_halos! — buffer-direct read for exact serial fold matching
#####

switch_north_halos!(c, north_bc, grid, loc, buffers) = nothing

function switch_north_halos!(c, north_bc::DistributedZipper, grid, loc, buffers)
    sign = north_bc.condition.sign
    fold_topo = grid.conformal_mapping.fold_topology
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    buf_x = buffer_x_interior(buffers.north, Hx, Nx)
    _switch_north_halos_from_buffer!(parent(c), buffers.north.recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x)

    # Fold-line half-row substitution: UPivot Center-y (CC, FC) and FPivot Face-y (CF, FF).
    # Serial kernels overwrite the right half (global i > Nx_global/2) of the fold line
    # with x-reversed values from the left half. In distributed:
    # - Multiple x-ranks: left rank keeps values, right rank replaces all from buffer.
    # - Single x-rank (Rx==1): only replace the right half (i > Nx÷2) to match serial.
    arch = architecture(grid)
    Rx = ranks(arch)[1]
    if Rx == 1
        _fold_line_from_buffer!(parent(c), buffers.north.recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x, Nx ÷ 2 + 1)
    elseif arch.local_index[1] > Rx ÷ 2
        _fold_line_from_buffer!(parent(c), buffers.north.recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x, 1)
    end

    # Fold-line x-halo corners (needed for Rx > 1).
    # The fold line's interior was just overwritten; corners must reflect accordingly.
    fold_line_y = _fold_line_parent_y(loc, Ny, Hy, fold_topo)
    if fold_line_y !== nothing
        fold_k = Hy + 2
        _fold_corner_write!(parent(c), sign, Hx, Nx, fold_line_y, fold_line_y,
                            buffers.north.recv, buf_x, fold_k, loc)
    end

    return nothing
end

# Buffer x-range dispatch (OneDBuffer has full x, TwoDBuffer has interior only)
@inline buffer_x_interior(::OneDBuffer, Hx, Nx) = Hx+1:Nx+Hx
@inline buffer_x_interior(::TwoDBuffer, Hx, Nx) = 1:Nx

# Fold parameters: (nj, dest_offset, src_k_fn) for each (fold_topo, y_location) pair.
# buffer[:, k, :] where k=1..Hy+2: k=1 = partner offset ny-Hy-1, k=Hy+2 = partner offset ny.
# Partner offset ny-m → buffer k = Hy+2-m.
#
# Pattern A (UPivot Face-y): nj=Hy, dest(j)=Ny+Hy+j, src_k(j)=Hy+3-j
# Pattern B (UPivot Center-y, FPivot Face-y): nj=Hy, dest(j)=Ny+Hy+j, src_k(j)=Hy+2-j
# Pattern C (FPivot Center-y): nj=Hy+1, dest(j)=Ny+Hy-1+j, src_k(j)=Hy+2-j

# Pattern A: UPivot Face-y (source: partner ny-j+1, dest: Ny+j)
@inline fold_nj(     ::Tuple{<:Any, <:Face,   <:Any}, Hy, ::Type{RightCenterFolded}) = Hy
@inline fold_dest_y( ::Tuple{<:Any, <:Face,   <:Any}, Ny, Hy, j, ::Type{RightCenterFolded}) = Ny + Hy + j
@inline fold_src_k(  ::Tuple{<:Any, <:Face,   <:Any}, Hy, j, ::Type{RightCenterFolded}) = Hy + 3 - j

# Pattern B: UPivot Center-y (source: partner ny-j, dest: Ny+j)
@inline fold_nj(     ::Tuple{<:Any, <:Center, <:Any}, Hy, ::Type{RightCenterFolded}) = Hy
@inline fold_dest_y( ::Tuple{<:Any, <:Center, <:Any}, Ny, Hy, j, ::Type{RightCenterFolded}) = Ny + Hy + j
@inline fold_src_k(  ::Tuple{<:Any, <:Center, <:Any}, Hy, j, ::Type{RightCenterFolded}) = Hy + 2 - j

# Pattern B: FPivot Face-y (same indexing as UPivot Center-y)
@inline fold_nj(     ::Tuple{<:Any, <:Face,   <:Any}, Hy, ::Type{RightFaceFolded}) = Hy
@inline fold_dest_y( ::Tuple{<:Any, <:Face,   <:Any}, Ny, Hy, j, ::Type{RightFaceFolded}) = Ny + Hy + j
@inline fold_src_k(  ::Tuple{<:Any, <:Face,   <:Any}, Hy, j, ::Type{RightFaceFolded}) = Hy + 2 - j

# Pattern C: FPivot Center-y (source: partner ny-j, dest: Ny-1+j, Hy+1 rows)
@inline fold_nj(     ::Tuple{<:Any, <:Center, <:Any}, Hy, ::Type{RightFaceFolded}) = Hy + 1
@inline fold_dest_y( ::Tuple{<:Any, <:Center, <:Any}, Ny, Hy, j, ::Type{RightFaceFolded}) = Ny + Hy - 1 + j
@inline fold_src_k(  ::Tuple{<:Any, <:Center, <:Any}, Hy, j, ::Type{RightFaceFolded}) = Hy + 2 - j

# x-reversed write for Center-x: standard reverse (i' = Nx+1-i)
@inline function _fold_x_write!(c, sign, Hx, Nx, dest_y, src_parent_y, north_recv, buf_x, src_k,
                                ::Tuple{<:Center, <:Any, <:Any})
    view(c, Hx+1:Hx+Nx, dest_y:dest_y, :) .= sign .* reverse(view(north_recv, buf_x, src_k:src_k, :), dims=1)
end

# x-reversed write for Face-x: cyclic shift of reverse (i' = Nx+2-i mod Nx)
# i=1 maps to i'=1 (fixed point) — use LOCAL parent, not partner buffer.
# i=2..Nx maps to i'=Nx..2 — use reversed partner buffer.
@inline function _fold_x_write!(c, sign, Hx, Nx, dest_y, src_parent_y, north_recv, buf_x, src_k,
                                ::Tuple{<:Face, <:Any, <:Any})
    view(c, Hx+1:Hx+1, dest_y:dest_y, :) .= sign .* view(c, Hx+1:Hx+1, src_parent_y:src_parent_y, :)
    view(c, Hx+2:Hx+Nx, dest_y:dest_y, :) .= sign .* reverse(view(north_recv, buf_x[2]:buf_x[end], src_k:src_k, :), dims=1)
end

# Corner writes for fold halo rows: fill x-halo columns at fold rows.
# For 2 equal-Nx x-ranks, the fold's x-reversal maps corner positions back to the
# SAME rank's interior columns (no additional MPI needed).

# Center-x corners: i' = Nx_global+1-i maps west halo to reversed local interior,
# and east halo to reversed local interior near the partition boundary.
@inline function _fold_corner_write!(c, sign, Hx, Nx, dest_y, src_parent_y, north_recv, buf_x, src_k,
                                     ::Tuple{<:Center, <:Any, <:Any})
    view(c, 1:Hx, dest_y:dest_y, :) .= sign .* reverse(view(c, Hx+1:2Hx, src_parent_y:src_parent_y, :), dims=1)
    view(c, Nx+Hx+1:Nx+2Hx, dest_y:dest_y, :) .= sign .* reverse(view(c, Nx+1:Nx+Hx, src_parent_y:src_parent_y, :), dims=1)
end

# Face-x corners: i' = Nx_global+2-i (shifted by 1 from Center-x, with periodic wrap).
# West halo reads from shifted interior. East halo k=1 needs partner's i=1 (from buffer),
# k=2:Hx reads from shifted local interior.
@inline function _fold_corner_write!(c, sign, Hx, Nx, dest_y, src_parent_y, north_recv, buf_x, src_k,
                                     ::Tuple{<:Face, <:Any, <:Any})
    view(c, 1:Hx, dest_y:dest_y, :) .= sign .* reverse(view(c, Hx+2:2Hx+1, src_parent_y:src_parent_y, :), dims=1)
    view(c, Nx+Hx+1:Nx+Hx+1, dest_y:dest_y, :) .= sign .* view(north_recv, buf_x[1]:buf_x[1], src_k:src_k, :)
    if Hx > 1
        view(c, Nx+Hx+2:Nx+2Hx, dest_y:dest_y, :) .= sign .* reverse(view(c, Nx+2:Nx+Hx, src_parent_y:src_parent_y, :), dims=1)
    end
end

# Fold-line half-row substitution from buffer for FPivot Face-y fields (CF, FF).
# Called only on right x-rank. Replaces local fold-line values with reversed partner data.
# Buffer k=Hy+2 holds the partner's fold-line row (partner parent y = Ny+Hy).
_fold_line_from_buffer!(c, recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x, fold_i_start) = nothing

# Center-x, Face-y, FPivot: reversal of fold line from partner buffer starting at fold_i_start
function _fold_line_from_buffer!(c, recv, ::Tuple{<:Center, <:Face, <:Any}, sign, Nx, Ny, Hx, Hy,
                                 ::Type{RightFaceFolded}, buf_x, fold_i_start)
    fold_y = Ny + Hy
    fold_k = Hy + 2  # buffer row for fold line
    reversed = sign .* reverse(view(recv, buf_x, fold_k:fold_k, :), dims=1)
    view(c, Hx+fold_i_start:Hx+Nx, fold_y:fold_y, :) .= view(reversed, fold_i_start:Nx, :, :)
end

# Face-x, Face-y, FPivot: i=1 pivot from local (when fold_i_start==1), i>=2 reversed from partner buffer
function _fold_line_from_buffer!(c, recv, ::Tuple{<:Face, <:Face, <:Any}, sign, Nx, Ny, Hx, Hy,
                                 ::Type{RightFaceFolded}, buf_x, fold_i_start)
    fold_y = Ny + Hy
    fold_k = Hy + 2
    if fold_i_start == 1
        view(c, Hx+1:Hx+1, fold_y:fold_y, :) .= sign .* view(c, Hx+1:Hx+1, fold_y:fold_y, :)
    end
    i_start = max(2, fold_i_start)
    reversed = sign .* reverse(view(recv, buf_x[2]:buf_x[end], fold_k:fold_k, :), dims=1)
    p_start = i_start - 1  # index into reversed buffer (which has Nx-1 elements)
    view(c, Hx+i_start:Hx+Nx, fold_y:fold_y, :) .= view(reversed, p_start:Nx-1, :, :)
end

# Center-x, Center-y, UPivot: same reversal pattern as Center-x, Face-y, FPivot
function _fold_line_from_buffer!(c, recv, ::Tuple{<:Center, <:Center, <:Any}, sign, Nx, Ny, Hx, Hy,
                                 ::Type{RightCenterFolded}, buf_x, fold_i_start)
    fold_y = Ny + Hy
    fold_k = Hy + 2
    reversed = sign .* reverse(view(recv, buf_x, fold_k:fold_k, :), dims=1)
    view(c, Hx+fold_i_start:Hx+Nx, fold_y:fold_y, :) .= view(reversed, fold_i_start:Nx, :, :)
end

# Face-x, Center-y, UPivot: i=1 pivot from local, i>=2 reversed from partner buffer
function _fold_line_from_buffer!(c, recv, ::Tuple{<:Face, <:Center, <:Any}, sign, Nx, Ny, Hx, Hy,
                                 ::Type{RightCenterFolded}, buf_x, fold_i_start)
    fold_y = Ny + Hy
    fold_k = Hy + 2
    if fold_i_start == 1
        view(c, Hx+1:Hx+1, fold_y:fold_y, :) .= sign .* view(c, Hx+1:Hx+1, fold_y:fold_y, :)
    end
    i_start = max(2, fold_i_start)
    reversed = sign .* reverse(view(recv, buf_x[2]:buf_x[end], fold_k:fold_k, :), dims=1)
    p_start = i_start - 1
    view(c, Hx+i_start:Hx+Nx, fold_y:fold_y, :) .= view(reversed, p_start:Nx-1, :, :)
end

# Fold-line parent y for cases that need fold-line overwrite; nothing otherwise.
_fold_line_parent_y(loc, Ny, Hy, fold_topo) = nothing
_fold_line_parent_y(::Tuple{<:Any, <:Center, <:Any}, Ny, Hy, ::Type{RightCenterFolded}) = Ny + Hy
_fold_line_parent_y(::Tuple{<:Any, <:Face, <:Any}, Ny, Hy, ::Type{RightFaceFolded}) = Ny + Hy

function _switch_north_halos_from_buffer!(c, north_recv, loc, sign, Nx, Ny, Hx, Hy, fold_topo, buf_x)
    nj = fold_nj(loc, Hy, fold_topo)

    # --- Interior x and corners for each fold halo row ---
    for j in 1:nj
        dest_y = fold_dest_y(loc, Ny, Hy, j, fold_topo)
        src_k  = fold_src_k(loc, Hy, j, fold_topo)
        # src_parent_y: local parent y corresponding to the same global row as buffer src_k.
        # Buffer k maps to partner parent y = Ny-1+(k-1) = Ny+k-2. Same for local rank.
        src_parent_y = Ny + src_k - 2
        _fold_x_write!(c, sign, Hx, Nx, dest_y, src_parent_y, north_recv, buf_x, src_k, loc)
        _fold_corner_write!(c, sign, Hx, Nx, dest_y, src_parent_y, north_recv, buf_x, src_k, loc)
    end

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
