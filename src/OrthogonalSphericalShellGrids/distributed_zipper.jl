using Oceananigans.BoundaryConditions: get_boundary_kernels, DistributedCommunication
using Oceananigans.DistributedComputations: cooperative_waitall!, recv_from_buffers!, distributed_fill_halo_event!
using Oceananigans.DistributedComputations: CommunicationBuffers, fill_corners!, loc_id, AsynchronousDistributed
using Oceananigans.Grids: AbstractGrid, topology,
    RightCenterFolded, RightFaceFolded,
    LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
    LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected
using Oceananigans.DistributedComputations: Distributed, on_architecture, ranks, x_communication_buffer
using Oceananigans.Fields: instantiated_location

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication!,
    y_communication_buffer, corner_communication_buffer,
    _fill_north_send_buffer!, _recv_from_north_buffer!,
    _fill_northwest_send_buffer!, _fill_northeast_send_buffer!,
    _recv_from_northwest_buffer!, _recv_from_northeast_buffer!,
    _fill_west_send_buffer!, _fill_east_send_buffer!,
    _recv_from_west_buffer!, _recv_from_east_buffer!
import Oceananigans.Fields: communication_buffers

@inline instantiate(T::DataType) = T()
@inline instantiate(T) = T

const DistributedZipper = BoundaryCondition{<:DistributedCommunication, <:ZipperHaloCommunicationRanks}

#####
##### Topology unions for dispatch
#####

const UPivotTopology = Union{RightCenterFolded,
                             LeftConnectedRightCenterFolded,
                             LeftConnectedRightCenterConnected}
const FPivotTopology = Union{RightFaceFolded,
                             LeftConnectedRightFaceFolded,
                             LeftConnectedRightFaceConnected}

# 1D fold (1xN, y-partitioned only) vs 2D fold (MxN, x+y partitioned)
const OneDFoldTopology = Union{LeftConnectedRightCenterFolded,
                               LeftConnectedRightFaceFolded}
const TwoDFoldTopology = Union{LeftConnectedRightCenterConnected,
                               LeftConnectedRightFaceConnected}

# UPivot fold line is at Center-y; FPivot fold line is at Face-y
has_fold_line(::Type{<:UPivotTopology}, ::Center) = true
has_fold_line(::Type{<:UPivotTopology}, ::Face)   = false
has_fold_line(::Type{<:FPivotTopology}, ::Center) = false
has_fold_line(::Type{<:FPivotTopology}, ::Face)   = true

# The fold has two pivot points due to x-periodicity:
#   1st pivot: between rx = Rx÷2 and rx = Rx÷2+1
#   2nd pivot: at the periodic boundary between rx = Rx and rx = 1
# The serial ifelse(i > Nx÷2, ...) writes the east half only.
# Corner conditions account for periodic wrap at rx=1 (NW) and rx=Rx (NE).

# North buffer: east-of-pivot ranks write fold line
function north_writes_fold_line(arch)
    rx = arch.local_index[1]
    Rx = ranks(arch)[1]
    return rx > Rx ÷ 2
end

# NW corner: west edge past pivot, or rx=1 (periodic wrap to east half)
function northwest_writes_fold_line(arch)
    rx = arch.local_index[1]
    Rx = ranks(arch)[1]
    return rx > Rx ÷ 2 + 1 || rx == 1
end

# NE corner: east edge past pivot, but not rx=Rx (periodic wrap to west half)
function northeast_writes_fold_line(arch)
    rx = arch.local_index[1]
    Rx = ranks(arch)[1]
    return rx >= Rx ÷ 2 && rx < Rx
end

#####
##### Zipper communication buffers with fold-aware packing
#####

struct OneDZipperBuffer{Loc, FoT, B, S}
    send :: B
    recv :: B
    sign :: S
end

struct TwoDZipperBuffer{Loc, FoT, B, S, FL, WFL}
    send :: B
    recv :: B
    sign :: S
end

struct ZipperCornerBuffer{Loc, FoT, B, S, FL, WFL}
    send :: B
    recv :: B
    sign :: S
end

# Value-argument constructors: all types inferred
OneDZipperBuffer(loc::Loc, fot::FoT, send::B, recv::B, sign::S) where {Loc, FoT, B, S} = OneDZipperBuffer{Loc, FoT, B, S}(send, recv, sign)
TwoDZipperBuffer(loc, fot, send, recv, sign, ::Val{FL}, ::Val{WFL}) where {FL, WFL} =
    TwoDZipperBuffer{typeof(loc), typeof(fot), typeof(send), typeof(sign), FL, WFL}(send, recv, sign)
ZipperCornerBuffer(loc, fot, send, recv, sign, ::Val{FL}, ::Val{WFL}) where {FL, WFL} =
    ZipperCornerBuffer{typeof(loc), typeof(fot), typeof(send), typeof(sign), FL, WFL}(send, recv, sign)
ZipperCornerBuffer(::Type{Loc}, ::Type{FoT}, send, recv, sign, ::Val{FL}, ::Val{WFL}) where {Loc, FoT, FL, WFL} =
    ZipperCornerBuffer{Loc, FoT, typeof(send), typeof(sign), FL, WFL}(send, recv, sign)

Adapt.adapt_structure(to, buff::OneDZipperBuffer) = nothing
Adapt.adapt_structure(to, buff::TwoDZipperBuffer) = nothing
Adapt.adapt_structure(to, buff::ZipperCornerBuffer) = nothing

# X-direction buffer for tripolar grids: like TwoDBuffer but with location-aware y-size.
# For fold-line fields on the north row, the x-buffer needs Ny+1 y-rows so that the
# fold-line row is exchanged periodically. Corners then overwrite where needed (WFL=true).
struct TripolarXBuffer{B}
    send :: B
    recv :: B
end

Adapt.adapt_structure(to, buff::TripolarXBuffer) = nothing

# TripolarXBuffer send/recv: use the buffer's actual y-size
function _fill_west_send_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny)
    @info "TripolarX west send: c x=$(1+Hx):$(2Hx), y=$(1+Hy):$(size(buff.send,2)+Hy), buff size=$(size(buff.send))"
    buff.send .= view(c, 1+Hx:2Hx, 1+Hy:size(buff.send,2)+Hy, :)
end
function _fill_east_send_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny)
    @info "TripolarX east send: c x=$(1+Nx):$(Nx+Hx), y=$(1+Hy):$(size(buff.send,2)+Hy), buff size=$(size(buff.send))"
    buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:size(buff.send,2)+Hy, :)
end
function _recv_from_west_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny)
    @info "TripolarX west recv: c x=1:$(Hx), y=$(1+Hy):$(size(buff.recv,2)+Hy), buff size=$(size(buff.recv))"
    view(c, 1:Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv
end
function _recv_from_east_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny)
    tgt_x, tgt_y = Nx+Hx+1, size(buff.recv,2)+Hy
    @info "TripolarX east recv: c x=$(1+Nx+Hx):$(Nx+2Hx), y=$(1+Hy):$(tgt_y), buff size=$(size(buff.recv)), c[$tgt_x,$tgt_y] BEFORE=$(c[tgt_x, tgt_y, (size(c,3)+1)÷2]), buff[1,end,1]=$(buff.recv[1, end, (size(buff.recv,3)+1)÷2])"
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv
    @info "TripolarX east recv: c[$tgt_x,$tgt_y] AFTER=$(c[tgt_x, tgt_y, (size(c,3)+1)÷2])"
end

# Fold-aware x-buffer: uniform Ny+1 for fold-line fields on north row.
# Fallback for non-zipper north (south ranks): standard x-communication.
x_tripolar_buffer(arch, grid, data, Hx, bc, loc, north) = x_communication_buffer(arch, grid, data, Hx, bc)

# When north IS a TwoDZipperBuffer (fold north row): use location-aware y-size
# (Ny+1 for BoundedTopology Face-y, i.e., FPivot CF/FF; Ny otherwise)
function x_tripolar_buffer(arch, grid, data, Hx, bc, loc,
                           north::TwoDZipperBuffer{Loc, FoT}) where {Loc, FoT}
    Ny_buf = length(Loc.parameters[2](), FoT(), size(grid, 2))
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    return TripolarXBuffer(send, recv)
end

#####
##### Buffer construction for tripolar grids
#####

function communication_buffers(grid::MPITripolarGridOfSomeKind, data, bcs, loc)
    Hx, Hy, Hz = halo_size(grid)
    arch = architecture(grid)

    south = y_communication_buffer(arch, grid, data, Hy, bcs.south)
    north = y_tripolar_buffer(arch, grid, data, Hy, bcs.north, loc)

    # x-buffers dispatch on north: Ny+1 for fold-line fields on north row
    west  = x_tripolar_buffer(arch, grid, data, Hx, bcs.west, loc, north)
    east  = x_tripolar_buffer(arch, grid, data, Hx, bcs.east, loc, north)

    @info "communication_buffers: rank=$(arch.local_index), loc=$loc, west=$(typeof(west)), east=$(typeof(east)), north=$(typeof(north))"

    sw = corner_communication_buffer(arch, grid, data, Hx, Hy, west, south)
    se = corner_communication_buffer(arch, grid, data, Hx, Hy, east, south)
    nw = northwest_tripolar_buffer(arch, grid, data, Hx, Hy, west, north)
    ne = northeast_tripolar_buffer(arch, grid, data, Hx, Hy, east, north)

    return CommunicationBuffers(west, east, south, north, sw, se, nw, ne)
end

# Fallback: non-zipper north BC uses standard buffer
y_tripolar_buffer(arch, grid, data, Hy, bc, loc) = y_communication_buffer(arch, grid, data, Hy, bc)

# 1D fold (1xN) → OneDZipperBuffer (full-width)
function y_tripolar_buffer(arch, grid::AbstractGrid{<:Any, <:Any, Topo},
                           data, Hy, bc::DistributedZipper, loc::Loc) where {Topo <: OneDFoldTopology, Loc}
    Tx, _, Tz = size(parent(data))
    FT = eltype(data)
    sgn = bc.condition.sign
    send = on_architecture(arch, zeros(FT, Tx, Hy, Tz))
    recv = on_architecture(arch, zeros(FT, Tx, Hy, Tz))
    return OneDZipperBuffer(loc, Topo(), send, recv, sgn)
end

# 2D fold (MxN) → TwoDZipperBuffer (interior-width, Hy′ = Hy or Hy+1 rows for fold line)
function y_tripolar_buffer(arch, grid::AbstractGrid{<:Any, <:Any, Topo},
                           data, Hy, bc::DistributedZipper, loc::Loc) where {Topo <: TwoDFoldTopology, Loc}
    Nx = size(grid, 1)
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    sgn = bc.condition.sign
    fl = has_fold_line(Topo, loc[2])
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && north_writes_fold_line(arch)
    send = on_architecture(arch, zeros(FT, Nx, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Nx, Hy′, Tz))
    return TwoDZipperBuffer(loc, Topo(), send, recv, sgn, Val(fl), Val(wfl))
end

# Fallbacks: non-zipper corners
northwest_tripolar_buffer(arch, grid, data, Hx, Hy, xedge, yedge) = corner_communication_buffer(arch, grid, data, Hx, Hy, xedge, yedge)
northeast_tripolar_buffer(arch, grid, data, Hx, Hy, xedge, yedge) = corner_communication_buffer(arch, grid, data, Hx, Hy, xedge, yedge)

# Corner buffers are only needed for 2D (MxN) partitions.
# FL (fold line in buffer) is true for ALL corners when has_fold_line, to ensure
# MPI size matching between mirror partners. WFL (writes fold line) is per-corner,
# based on whether the corner's global x-position is past the pivot.

# ── NW corner: Center-x (Hx columns) ──
function northwest_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{Loc, FoT}) where {Loc <: Tuple{<:Center, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    fl = has_fold_line(FoT, Loc.parameters[2]())
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northwest_writes_fold_line(arch)
    send = on_architecture(arch, zeros(FT, Hx, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Hy′, Tz))
    return ZipperCornerBuffer(Loc, FoT, send, recv, sgn, Val(fl), Val(wfl))
end

# ── NW corner: Face-x (Hx+1 columns for the Face-x shift) ──
function northwest_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{Loc, FoT}) where {Loc <: Tuple{<:Face, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    fl = has_fold_line(FoT, Loc.parameters[2]())
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northwest_writes_fold_line(arch)
    send = on_architecture(arch, zeros(FT, Hx+1, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx+1, Hy′, Tz))
    return ZipperCornerBuffer(Loc, FoT, send, recv, sgn, Val(fl), Val(wfl))
end

# ── NE corner: Center-x (Hx columns) ──
function northeast_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{Loc, FoT}) where {Loc <: Tuple{<:Center, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    fl = has_fold_line(FoT, Loc.parameters[2]())
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northeast_writes_fold_line(arch)
    send = on_architecture(arch, zeros(FT, Hx, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Hy′, Tz))
    return ZipperCornerBuffer(Loc, FoT, send, recv, sgn, Val(fl), Val(wfl))
end

# ── NE corner: Face-x (Hx-1 columns, north buffer covers the extra Face-x column) ──
function northeast_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{Loc, FoT}) where {Loc <: Tuple{<:Face, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    fl = has_fold_line(FoT, Loc.parameters[2]())
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northeast_writes_fold_line(arch)
    send = on_architecture(arch, zeros(FT, Hx - 1, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx - 1, Hy′, Tz))
    return ZipperCornerBuffer(Loc, FoT, send, recv, sgn, Val(fl), Val(wfl))
end

#####
##### Location aliases for dispatch
#####

const CC = Tuple{<:Center, <:Center, <:Any}
const FC = Tuple{<:Face,   <:Center, <:Any}
const CF = Tuple{<:Center, <:Face,   <:Any}
const FF = Tuple{<:Face,   <:Face,   <:Any}

#####
##### Fold-aware send buffer packing: _fill_north_send_buffer!
#####
##### Y-range rules (parent-array coords, Ny = size(grid, 2)):
#####   UPivot Center-y (CC/FC): skip fold at Ny → Hy rows from Ny-1: Ny+Hy-1:-1:Ny
#####   All other cases:         Hy rows from Ny:                     Ny+Hy:-1:Ny+1
#####
##### For TwoD fold-line fields (Hy+1 buffer): row 1 = fold line, rows 2:Hy+1 = halo sources
#####

# ── TwoDZipperBuffer: UPivot Center-y (CC/FC) — fold line + Hy halo rows ──

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CC UPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y fold=$(Ny+Hy), y halo=$(Ny)..$(Ny+Hy-1), buff size=$(size(b.send))"
    view(b.send, :, 1:1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:Ny+Hy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy-1:-1:Ny, :)
end

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FC UPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y fold=$(Ny+Hy), y halo=$(Ny)..$(Ny+Hy-1), buff size=$(size(b.send)), c[$(1+Hx),$(Ny+Hy-1),1]=$(c[1+Hx, Ny+Hy-1, (size(c,3)+1)÷2])"
    view(b.send, :, 1:1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:Ny+Hy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy-1:-1:Ny, :)
    @info "FC UPivot north send: buff[end,2,1]=$(b.send[end, 2, (size(b.send,3)+1)÷2]) (should go to mirror's last x at first halo row)"
end

# ── TwoDZipperBuffer: UPivot Face-y (CF/FF) — no fold line, Hy halo rows ──

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CF UPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y=$(Ny+1)..$(Ny+Hy), buff size=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end
function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FF UPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y=$(Ny+1)..$(Ny+Hy), buff size=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end

# ── TwoDZipperBuffer: FPivot Center-y (CC/FC) — no fold line, Hy halo rows ──

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CC FPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y=$(Ny+1)..$(Ny+Hy), buff size=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end
function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FC FPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y=$(Ny+1)..$(Ny+Hy), buff size=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end

# ── TwoDZipperBuffer: FPivot Face-y (CF/FF) — fold line + Hy halo rows ──

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CF FPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y fold=$(Ny+1+Hy), y halo=$(Ny+1)..$(Ny+Hy), buff size=$(size(b.send))"
    view(b.send, :, 1:1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+1+Hy:Ny+1+Hy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FF FPivot north send: x=$(1+Hx):$(Nx+Hx) (reversed), y fold=$(Ny+1+Hy), y halo=$(Ny+1)..$(Ny+Hy), buff size=$(size(b.send))"
    view(b.send, :, 1:1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+1+Hy:Ny+1+Hy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end

# ── OneDZipperBuffer: UPivot Center-y (CC/FC) — simple reverse ──

_fill_north_send_buffer!(c, b::OneDZipperBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, size(c,1):-1:1, Ny+Hy-1:-1:Ny, :)

function _fill_north_send_buffer!(c, b::OneDZipperBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    Tx = size(c, 1)
    view(b.send, 2:Tx, :, :) .= b.sign .* view(c, Tx:-1:2, Ny+Hy-1:-1:Ny, :)
    view(b.send, 1:1,  :, :) .= b.sign .* view(c, 1:1,     Ny+Hy-1:-1:Ny, :)
end

# ── OneDZipperBuffer: UPivot Face-y (CF/FF) ──

_fill_north_send_buffer!(c, b::OneDZipperBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, size(c,1):-1:1, Ny+Hy:-1:Ny+1, :)

function _fill_north_send_buffer!(c, b::OneDZipperBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    Tx = size(c, 1)
    view(b.send, 2:Tx, :, :) .= b.sign .* view(c, Tx:-1:2, Ny+Hy:-1:Ny+1, :)
    view(b.send, 1:1,  :, :) .= b.sign .* view(c, 1:1,     Ny+Hy:-1:Ny+1, :)
end

# ── OneDZipperBuffer: FPivot Center-y (CC/FC) ──

_fill_north_send_buffer!(c, b::OneDZipperBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, size(c,1):-1:1, Ny+Hy:-1:Ny+1, :)

function _fill_north_send_buffer!(c, b::OneDZipperBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    Tx = size(c, 1)
    view(b.send, 2:Tx, :, :) .= b.sign .* view(c, Tx:-1:2, Ny+Hy:-1:Ny+1, :)
    view(b.send, 1:1,  :, :) .= b.sign .* view(c, 1:1,     Ny+Hy:-1:Ny+1, :)
end

# ── OneDZipperBuffer: FPivot Face-y (CF/FF) ──

_fill_north_send_buffer!(c, b::OneDZipperBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, size(c,1):-1:1, Ny+Hy:-1:Ny+1, :)

function _fill_north_send_buffer!(c, b::OneDZipperBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    Tx = size(c, 1)
    view(b.send, 2:Tx, :, :) .= b.sign .* view(c, Tx:-1:2, Ny+Hy:-1:Ny+1, :)
    view(b.send, 1:1,  :, :) .= b.sign .* view(c, 1:1,     Ny+Hy:-1:Ny+1, :)
end

#####
##### Fold-aware send buffer packing: NW corner (leftmost Hx interior columns)
#####

# Center-x NW: Hy rows, or Hy+1 when fold line is at this y-location
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CC UPivot NW send: x=$(1+Hx):$(2Hx) (rev), y=$(Ny)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+Hy:-1:Ny, :)
end
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CC FPivot NW send: x=$(1+Hx):$(2Hx) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CF UPivot NW send: x=$(1+Hx):$(2Hx) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CF FPivot NW send: x=$(1+Hx):$(2Hx) (rev), y=$(Ny+1)..$(Ny+1+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+1+Hy:-1:Ny+1, :)
end

# Face-x NW: Hx+1 columns from leftmost Hx+1 interior columns
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FC UPivot NW send: x=$(1+Hx):$(2Hx+1) (rev), y=$(Ny)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+Hy:-1:Ny, :)
end
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FC FPivot NW send: x=$(1+Hx):$(2Hx+1) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FF UPivot NW send: x=$(1+Hx):$(2Hx+1) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end
function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FF FPivot NW send: x=$(1+Hx):$(2Hx+1) (rev), y=$(Ny+1)..$(Ny+1+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+1+Hy:-1:Ny+1, :)
end

#####
##### Fold-aware send buffer packing: NE corner (rightmost Hx interior columns)
#####

# Center-x NE: Hy rows, or Hy+1 when fold line is at this y-location
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CC UPivot NE send: x=$(1+Nx):$(Nx+Hx) (rev), y=$(Ny)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+Hy:-1:Ny, :)
end
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CC FPivot NE send: x=$(1+Nx):$(Nx+Hx) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+Hy:-1:Ny+1, :)
end
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CF UPivot NE send: x=$(1+Nx):$(Nx+Hx) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+Hy:-1:Ny+1, :)
end
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "CF FPivot NE send: x=$(1+Nx):$(Nx+Hx) (rev), y=$(Ny+1)..$(Ny+1+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+1+Hy:-1:Ny+1, :)
end

# Face-x NE: Hx-1 columns from rightmost Hx-1 interior columns
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FC UPivot NE send: x=$(Nx+2):$(Nx+Hx) (rev), y=$(Ny)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+Hy:-1:Ny, :)
end
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FC FPivot NE send: x=$(Nx+2):$(Nx+Hx) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+Hy:-1:Ny+1, :)
end
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FF UPivot NE send: x=$(Nx+2):$(Nx+Hx) (rev), y=$(Ny+1)..$(Ny+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+Hy:-1:Ny+1, :)
end
function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    @info "FF FPivot NE send: x=$(Nx+2):$(Nx+Hx) (rev), y=$(Ny+1)..$(Ny+1+Hy), buff=$(size(b.send))"
    b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+1+Hy:-1:Ny+1, :)
end

#####
##### Recv methods: direct placement (data is already folded by sender)
#####
##### For TwoD fold-line fields (FL=true, Hy+1 buffer):
#####   WFL=true:  write fold line (row 1) + halos (rows 2:Hy+1)
#####   WFL=false: write halos only (rows 2:Hy+1, skip fold line)
#####
##### For FPivot Face-y: halos start at parent Ny+Hy+2 (field Ny+2, past fold line at Ny+1)
#####

# ── TwoDZipperBuffer: UPivot CC/FC — FL=true, dispatch on WFL ──

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CC, <:LeftConnectedRightCenterConnected, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    @info "CC UPivot north recv WFL=true: x=$(1+Hx):$(Nx+Hx), y=$(Ny+Hy):$(Ny+2Hy), buff size=$(size(buff.recv))"
    view(c, 1+Hx:Nx+Hx, Ny+Hy:Ny+Hy, :)   .= view(buff.recv, :, 1:1, :)
    view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CC, <:LeftConnectedRightCenterConnected, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny)
    @info "CC UPivot north recv WFL=false: x=$(1+Hx):$(Nx+Hx), y=$(1+Ny+Hy):$(Ny+2Hy), buff size=$(size(buff.recv))"
    view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FC, <:LeftConnectedRightCenterConnected, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    @info "FC UPivot north recv WFL=true: x=$(2+Hx):$(Nx+Hx+1), y=$(Ny+Hy):$(Ny+2Hy), buff size=$(size(buff.recv))"
    view(c, 2+Hx:Nx+Hx+1, Ny+Hy:Ny+Hy, :)   .= view(buff.recv, :, 1:1, :)
    view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FC, <:LeftConnectedRightCenterConnected, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny)
    tgt_x, tgt_y, tgt_z = Nx+Hx+1, 1+Ny+Hy, (size(c,3)+1)÷2
    @info "FC UPivot north recv WFL=false: x=$(2+Hx):$(Nx+Hx+1), y=$(1+Ny+Hy):$(Ny+2Hy), c[$tgt_x,$tgt_y,$tgt_z] BEFORE=$(c[tgt_x, tgt_y, tgt_z]), buff[end,2,$tgt_z]=$(buff.recv[end, 2, tgt_z])"
    view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
    @info "FC UPivot north recv WFL=false: c[$tgt_x,$tgt_y,$tgt_z] AFTER=$(c[tgt_x, tgt_y, tgt_z])"
end

# ── TwoDZipperBuffer: UPivot CF/FF — FL=false, no fold line ──

_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── TwoDZipperBuffer: FPivot CC/FC — FL=false, no fold line ──

_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── TwoDZipperBuffer: FPivot CF/FF — FL=true, dispatch on WFL, shifted +1 for Face-y ──

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CF, <:LeftConnectedRightFaceConnected, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 1+Hx:Nx+Hx, Ny+1+Hy:Ny+1+Hy, :) .= view(buff.recv, :, 1:1, :)
    view(c, 1+Hx:Nx+Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CF, <:LeftConnectedRightFaceConnected, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny)
    view(c, 1+Hx:Nx+Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FF, <:LeftConnectedRightFaceConnected, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 2+Hx:Nx+Hx+1, Ny+1+Hy:Ny+1+Hy, :) .= view(buff.recv, :, 1:1, :)
    view(c, 2+Hx:Nx+Hx+1, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FF, <:LeftConnectedRightFaceConnected, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny)
    view(c, 2+Hx:Nx+Hx+1, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

# ── OneDZipperBuffer: UPivot — standard placement ──

_recv_from_north_buffer!(c, buff::OneDZipperBuffer{<:Any, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, :, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── OneDZipperBuffer: FPivot CC/FC — standard placement ──

_recv_from_north_buffer!(c, buff::OneDZipperBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, :, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_north_buffer!(c, buff::OneDZipperBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, :, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── OneDZipperBuffer: FPivot CF/FF — shifted +1 for Face-extended ──

_recv_from_north_buffer!(c, buff::OneDZipperBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, :, 2+Ny+Hy:1+Ny+2Hy, :) .= buff.recv
_recv_from_north_buffer!(c, buff::OneDZipperBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, :, 2+Ny+Hy:1+Ny+2Hy, :) .= buff.recv

# ── Corner recv: UPivot CC/FC — FL=true, dispatch on WFL ──

# WFL=true: write fold line + halos
function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:UPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 1:Hx, Ny+Hy:Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end
function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:UPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 1:Hx+1, Ny+Hy:Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 1:Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end
function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:UPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 1+Nx+Hx:Nx+2Hx, Ny+Hy:Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end
function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:UPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    @info "FC UPivot NE corner recv WFL=true: x=$(2+Nx+Hx):$(Nx+2Hx), y=$(Ny+Hy):$(Ny+2Hy), buff size=$(size(buff.recv))"
    view(c, 2+Nx+Hx:Nx+2Hx, Ny+Hy:Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

# WFL=false: write halos only (skip fold line)
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:UPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:UPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:UPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:UPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny)
    @info "FC UPivot NE corner recv WFL=false: x=$(2+Nx+Hx):$(Nx+2Hx), y=$(1+Ny+Hy):$(Ny+2Hy), buff size=$(size(buff.recv))"
    view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

# ── Corner recv: UPivot CF/FF — FL=false, no fold line, Hy rows ──

_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── Corner recv: FPivot CC/FC — FL=false, no fold line, Hy rows ──

_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── Corner recv: FPivot CF/FF — FL=true, dispatch on WFL, shifted +1 for Face-y ──

# WFL=true: write fold line + halos
function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:FPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 1:Hx, 1+Ny+Hy:1+Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 1:Hx, 2+Ny+Hy:1+Ny+2Hy, :)    .= view(buff.recv, :, 2:size(buff.recv,2), :)
end
function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:FPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 1:Hx+1, 1+Ny+Hy:1+Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 1:Hx+1, 2+Ny+Hy:1+Ny+2Hy, :)    .= view(buff.recv, :, 2:size(buff.recv,2), :)
end
function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:FPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:1+Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 1+Nx+Hx:Nx+2Hx, 2+Ny+Hy:1+Ny+2Hy, :)    .= view(buff.recv, :, 2:size(buff.recv,2), :)
end
function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:FPivotTopology, <:Any, <:Any, true, true}, Hx, Hy, Nx, Ny)
    view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:1+Ny+Hy, :)    .= view(buff.recv, :, 1:1, :)
    view(c, 2+Nx+Hx:Nx+2Hx, 2+Ny+Hy:1+Ny+2Hy, :)    .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

# WFL=false: write halos only (skip fold line)
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:FPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:FPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:FPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:FPivotTopology, <:Any, <:Any, true, false}, Hx, Hy, Nx, Ny) = view(c, 2+Nx+Hx:Nx+2Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)

#####
##### _switch_north_halos! — no-op (fold logic in send buffers)
#####

switch_north_halos!(c, north_bc, grid, loc) = nothing

#####
##### fill_halo_regions! for distributed tripolar grids
#####

fill_halo_regions!(c::OffsetArray, ::Nothing, indices, loc, ::MPITripolarGridOfSomeKind, args...; kwargs...) = nothing

function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::MPITripolarGridOfSomeKind, buffers::CommunicationBuffers, args...; kwargs...)

    arch = architecture(grid)
    kernels!, ordered_bcs = get_boundary_kernels(bcs, c, grid, loc, indices)

    number_of_tasks = length(kernels!)
    outstanding_requests = length(arch.mpi_requests)

    for task = 1:number_of_tasks
        @inbounds distributed_fill_halo_event!(c, kernels![task], ordered_bcs[task], loc, grid, buffers, args...; kwargs...)
    end

    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers, args...; kwargs...)

    if length(arch.mpi_requests) > outstanding_requests
        arch.mpi_tag[] += 1
    end

    return nothing
end

function synchronize_communication!(field::Field{<:Any, <:Any, <:Any, <:Any, <:MPITripolarGridOfSomeKind})
    arch = architecture(field.grid)

    if arch isa AsynchronousDistributed
        if !isempty(arch.mpi_requests)
            cooperative_waitall!(arch.mpi_requests)
            arch.mpi_tag[] = 0
            empty!(arch.mpi_requests)
        end

        recv_from_buffers!(field.data, field.communication_buffers, field.grid)
    end

    return nothing
end
