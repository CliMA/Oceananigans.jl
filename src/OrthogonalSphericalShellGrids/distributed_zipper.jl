using Oceananigans.BoundaryConditions: get_boundary_kernels, DistributedCommunication
using Oceananigans.DistributedComputations: cooperative_waitall!, recv_from_buffers!, distributed_fill_halo_event!
using Oceananigans.DistributedComputations: CommunicationBuffers, fill_corners!, loc_id, AsynchronousDistributed
using Oceananigans.Grids: AbstractGrid, topology,
    RightCenterFolded, RightFaceFolded,
    LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
    LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected,
    WestOfPivot, EastOfPivot
using Oceananigans.DistributedComputations: Distributed, on_architecture, ranks, x_communication_buffer
using Oceananigans.Fields: instantiated_location

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication!,
    y_communication_buffer, corner_communication_buffer,
    _fill_north_send_buffer!, _recv_from_north_buffer!,
    _fill_northwest_send_buffer!, _fill_northeast_send_buffer!,
    _recv_from_northwest_buffer!, _recv_from_northeast_buffer!
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

#####
##### Zipper communication buffers with fold-aware packing
#####

struct OneDZipperBuffer{Loc, FoT, B, S}
    send :: B
    recv :: B
    sign :: S
end

struct TwoDZipperBuffer{Loc, FoT, B, S}
    send :: B
    recv :: B
    sign :: S
end

struct ZipperCornerBuffer{Loc, FoT, B, S}
    send :: B
    recv :: B
    sign :: S
end

# Value-argument constructors: all types inferred
OneDZipperBuffer(loc::Loc, fot::FoT, send::B, recv::B, sign::S) where {Loc, FoT, B, S} = OneDZipperBuffer{Loc, FoT, B, S}(send, recv, sign)
TwoDZipperBuffer(loc::Loc, fot::FoT, send::B, recv::B, sign::S) where {Loc, FoT, B, S} = TwoDZipperBuffer{Loc, FoT, B, S}(send, recv, sign)
ZipperCornerBuffer(loc::Loc, fot::FoT, send::B, recv::B, sign::S) where {Loc, FoT, B, S} = ZipperCornerBuffer{Loc, FoT, B, S}(send, recv, sign)

Adapt.adapt_structure(to, buff::OneDZipperBuffer) = nothing
Adapt.adapt_structure(to, buff::TwoDZipperBuffer) = nothing
Adapt.adapt_structure(to, buff::ZipperCornerBuffer) = nothing

#####
##### Buffer construction for tripolar grids
#####

function communication_buffers(grid::MPITripolarGridOfSomeKind, data, bcs, loc)
    Hx, Hy, Hz = halo_size(grid)
    arch = architecture(grid)

    west  = x_communication_buffer(arch, grid, data, Hx, bcs.west)
    east  = x_communication_buffer(arch, grid, data, Hx, bcs.east)
    south = y_communication_buffer(arch, grid, data, Hy, bcs.south)
    north = y_tripolar_buffer(arch, grid, data, Hy, bcs.north, loc)

    sw = corner_communication_buffer(arch, grid, data, Hx, Hy, west, south)
    se = corner_communication_buffer(arch, grid, data, Hx, Hy, east, south)
    nw = northwest_tripolar_buffer(arch, grid, data, Hx, Hy, west, north)
    ne = northeast_tripolar_buffer(arch, grid, data, Hx, Hy, east, north)

    return CommunicationBuffers(west, east, south, north, sw, se, nw, ne)
end

# Fallback: non-zipper north BC uses standard buffer
y_tripolar_buffer(arch, grid, data, H, bc, loc) = y_communication_buffer(arch, grid, data, H, bc)

# 1D fold (1xN) → OneDZipperBuffer (full-width)
function y_tripolar_buffer(arch::Distributed, grid::AbstractGrid{<:Any, <:Any, Topo},
                           data, H, bc::DistributedZipper, loc::Loc) where {Topo <: OneDFoldTopology, Loc}
    Tx, _, Tz = size(parent(data))
    FT = eltype(data)
    sgn = bc.condition.sign
    send = on_architecture(arch, zeros(FT, Tx, H, Tz))
    recv = on_architecture(arch, zeros(FT, Tx, H, Tz))
    return OneDZipperBuffer(loc, Topo(), send, recv, sgn)
end

# 2D fold (MxN) → TwoDZipperBuffer (interior-width, Hy or Hy+1 rows for fold line)
function y_tripolar_buffer(arch::Distributed, grid::AbstractGrid{<:Any, <:Any, Topo},
                           data, H, bc::DistributedZipper, loc::Loc) where {Topo <: TwoDFoldTopology, Loc}
    Nx = size(grid, 1)
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    sgn = bc.condition.sign
    Hbuf = has_fold_line(Topo, loc[2]) ? H + 1 : H
    send = on_architecture(arch, zeros(FT, Nx, Hbuf, Tz))
    recv = on_architecture(arch, zeros(FT, Nx, Hbuf, Tz))
    return TwoDZipperBuffer(loc, Topo(), send, recv, sgn)
end

# Fallbacks: non-zipper corners
northwest_tripolar_buffer(arch, grid, data, Hx, Hy, xedge, yedge) = corner_communication_buffer(arch, grid, data, Hx, Hy, xedge, yedge)
northeast_tripolar_buffer(arch, grid, data, Hx, Hy, xedge, yedge) = corner_communication_buffer(arch, grid, data, Hx, Hy, xedge, yedge)

# ── NW corner: Center-x (Hx columns) ──
function northwest_tripolar_buffer(arch::Distributed, grid, data, Hx, Hy, xedge,
                                   yedge::Union{<:OneDZipperBuffer{Loc, FoT},
                                                <:TwoDZipperBuffer{Loc, FoT}}) where {Loc <: Tuple{<:Center, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    send = on_architecture(arch, zeros(FT, Hx, Hy, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Hy, Tz))
    return ZipperCornerBuffer{Loc, FoT, typeof(send), typeof(sgn)}(send, recv, sgn)
end

# ── NW corner: Face-x (Hx+1 columns for the Face-x shift) ──
function northwest_tripolar_buffer(arch::Distributed, grid, data, Hx, Hy, xedge,
                                   yedge::Union{<:OneDZipperBuffer{Loc, FoT},
                                                <:TwoDZipperBuffer{Loc, FoT}}) where {Loc <: Tuple{<:Face, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    send = on_architecture(arch, zeros(FT, Hx+1, Hy, Tz))
    recv = on_architecture(arch, zeros(FT, Hx+1, Hy, Tz))
    return ZipperCornerBuffer{Loc, FoT, typeof(send), typeof(sgn)}(send, recv, sgn)
end

# ── NE corner: Center-x (Hx columns) ──
function northeast_tripolar_buffer(arch::Distributed, grid, data, Hx, Hy, xedge,
                                   yedge::Union{<:OneDZipperBuffer{Loc, FoT},
                                                <:TwoDZipperBuffer{Loc, FoT}}) where {Loc <: Tuple{<:Center, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    send = on_architecture(arch, zeros(FT, Hx, Hy, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Hy, Tz))
    return ZipperCornerBuffer{Loc, FoT, typeof(send), typeof(sgn)}(send, recv, sgn)
end

# ── NE corner: Face-x (Hx-1 columns, TwoDBuffer covers the extra) ──
function northeast_tripolar_buffer(arch::Distributed, grid, data, Hx, Hy, xedge,
                                   yedge::Union{<:OneDZipperBuffer{Loc, FoT},
                                                <:TwoDZipperBuffer{Loc, FoT}}) where {Loc <: Tuple{<:Face, <:Any, <:Any}, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    Hx_ne = max(Hx - 1, 0)
    send = on_architecture(arch, zeros(FT, Hx_ne, Hy, Tz))
    recv = on_architecture(arch, zeros(FT, Hx_ne, Hy, Tz))
    return ZipperCornerBuffer{Loc, FoT, typeof(send), typeof(sgn)}(send, recv, sgn)
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
    view(b.send, :, 1:1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:Ny+Hy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy-1:-1:Ny, :)
end

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny)
    view(b.send, :, 1:1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:Ny+Hy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy-1:-1:Ny, :)
end

# ── TwoDZipperBuffer: UPivot Face-y (CF/FF) — no fold line, Hy halo rows ──

_fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
_fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)

# ── TwoDZipperBuffer: FPivot Center-y (CC/FC) — no fold line, Hy halo rows ──

_fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
_fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)

# ── TwoDZipperBuffer: FPivot Face-y (CF/FF) — fold line + Hy halo rows ──

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
    view(b.send, :, 1:1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+1+Hy:Ny+1+Hy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
end

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny)
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

# Center-x NW
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+Hy-1:-1:Ny, :)
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx:-1:1+Hx, Ny+Hy:-1:Ny+1, :)

# Face-x NW: Hx+1 columns from leftmost Hx+1 interior columns
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+Hy-1:-1:Ny, :)
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+Hy:-1:Ny+1, :)
_fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, 2Hx+1:-1:1+Hx, Ny+Hy:-1:Ny+1, :)

#####
##### Fold-aware send buffer packing: NE corner (rightmost Hx interior columns)
#####

# Center-x NE
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+Hy-1:-1:Ny, :)
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+Hy:-1:Ny+1, :)
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+Hy:-1:Ny+1, :)
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:1+Nx, Ny+Hy:-1:Ny+1, :)

# Face-x NE: Hx-1 columns from rightmost Hx-1 interior columns
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+Hy-1:-1:Ny, :)
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+Hy:-1:Ny+1, :)
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+Hy:-1:Ny+1, :)
_fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = b.send .= b.sign .* view(c, Nx+Hx:-1:Nx+2, Ny+Hy:-1:Ny+1, :)

#####
##### Recv methods: direct placement (data is already folded by sender)
#####
##### For TwoD fold-line fields (Hy+1 buffer):
#####   EastOfPivot: write fold line + halos
#####   WestOfPivot: write halos only (skip fold line in row 1)
#####
##### For FPivot Face-y: halos start at parent Ny+Hy+2 (field Ny+2, past fold line at Ny+1)
#####

# ── TwoDZipperBuffer: UPivot CC/FC — fold line dispatch on WoP/EoP ──

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CC, <:LeftConnectedRightCenterConnected{EastOfPivot}}, Hx, Hy, Nx, Ny)
    view(c, 1+Hx:Nx+Hx, Ny+Hy:Ny+Hy, :)   .= view(buff.recv, :, 1:1, :)
    view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CC, <:LeftConnectedRightCenterConnected{WestOfPivot}}, Hx, Hy, Nx, Ny)
    view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FC, <:LeftConnectedRightCenterConnected{EastOfPivot}}, Hx, Hy, Nx, Ny)
    view(c, 2+Hx:Nx+Hx+1, Ny+Hy:Ny+Hy, :)   .= view(buff.recv, :, 1:1, :)
    view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FC, <:LeftConnectedRightCenterConnected{WestOfPivot}}, Hx, Hy, Nx, Ny)
    view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

# ── TwoDZipperBuffer: UPivot CF/FF — no fold line, standard placement ──

_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── TwoDZipperBuffer: FPivot CC/FC — no fold line, standard placement ──

_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Hx:Nx+Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Hx:Nx+Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── TwoDZipperBuffer: FPivot CF/FF — fold line dispatch, shifted +1 for Face-extended ──

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CF, <:LeftConnectedRightFaceConnected{EastOfPivot}}, Hx, Hy, Nx, Ny)
    view(c, 1+Hx:Nx+Hx, Ny+1+Hy:Ny+1+Hy, :) .= view(buff.recv, :, 1:1, :)
    view(c, 1+Hx:Nx+Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:CF, <:LeftConnectedRightFaceConnected{WestOfPivot}}, Hx, Hy, Nx, Ny)
    view(c, 1+Hx:Nx+Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FF, <:LeftConnectedRightFaceConnected{EastOfPivot}}, Hx, Hy, Nx, Ny)
    view(c, 2+Hx:Nx+Hx+1, Ny+1+Hy:Ny+1+Hy, :) .= view(buff.recv, :, 1:1, :)
    view(c, 2+Hx:Nx+Hx+1, 2+Ny+Hy:1+Ny+2Hy, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{<:FF, <:LeftConnectedRightFaceConnected{WestOfPivot}}, Hx, Hy, Nx, Ny)
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

# ── Corner recv: UPivot — standard placement ──

_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:UPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── Corner recv: FPivot CC/FC — standard placement ──

_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FC, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Nx+Hx:Nx+2Hx, 1+Ny+Hy:Ny+2Hy, :) .= buff.recv

# ── Corner recv: FPivot CF/FF — shifted +1 for Face-extended ──

_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= buff.recv
_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1:Hx+1, 2+Ny+Hy:1+Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:CF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 1+Nx+Hx:Nx+2Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= buff.recv
_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{<:FF, <:FPivotTopology}, Hx, Hy, Nx, Ny) = view(c, 2+Nx+Hx:Nx+2Hx, 2+Ny+Hy:1+Ny+2Hy, :) .= buff.recv

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
