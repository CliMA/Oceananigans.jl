using Oceananigans.BoundaryConditions: DistributedCommunication
using Oceananigans.DistributedComputations: CommunicationBuffers, loc_id
using Oceananigans.Grids: AbstractGrid, topology,
    RightCenterFolded, RightFaceFolded,
    LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
    LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected,
    PencilFoldedTopology
using Oceananigans.DistributedComputations: Distributed, on_architecture, ranks, x_communication_buffer

import Oceananigans.DistributedComputations:
    y_communication_buffer, corner_communication_buffer,
    _fill_north_send_buffer!, _recv_from_north_buffer!,
    _fill_northwest_send_buffer!, _fill_northeast_send_buffer!,
    _recv_from_northwest_buffer!, _recv_from_northeast_buffer!,
    _fill_west_send_buffer!, _fill_east_send_buffer!,
    _recv_from_west_buffer!, _recv_from_east_buffer!
import Oceananigans.Fields: communication_buffers

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

struct TwoDZipperBuffer{FL, WFL, TY, Loc, B, S}
    loc  :: Loc
    send :: B
    recv :: B
    sign :: S
end

struct ZipperCornerBuffer{FL, WFL, TY, Loc, B, S}
    loc  :: Loc
    send :: B
    recv :: B
    sign :: S
end

# Partial type-parameter constructors: FL, WFL, TY specified; Loc, B, S inferred
TwoDZipperBuffer{FL, WFL, TY}(loc::Loc, send::B, recv::B, sign::S) where {FL, WFL, TY, Loc, B, S} =
    TwoDZipperBuffer{FL, WFL, TY, Loc, B, S}(loc, send, recv, sign)
ZipperCornerBuffer{FL, WFL, TY}(loc::Loc, send::B, recv::B, sign::S) where {FL, WFL, TY, Loc, B, S} =
    ZipperCornerBuffer{FL, WFL, TY, Loc, B, S}(loc, send, recv, sign)

Adapt.adapt_structure(to, buff::TwoDZipperBuffer) = nothing
Adapt.adapt_structure(to, buff::ZipperCornerBuffer) = nothing

# X-direction buffer for tripolar grids: like TwoDBuffer but with location-aware y-size
# and fold-line awareness. FL/WFL mirror the corner buffer pattern:
# - FL: buffer has fold-line row (same as corners, for MPI size matching)
# - WFL: recv writes the fold-line row (complement of the adjacent corner)
#   West WFL = !NW corner WFL, East WFL = !NE corner WFL.
struct TripolarXBuffer{FL, WFL, B}
    send :: B
    recv :: B
end

Adapt.adapt_structure(to, buff::TripolarXBuffer) = nothing

TripolarXBuffer{FL, WFL}(send::B, recv::B) where {FL, WFL, B} =
    TripolarXBuffer{FL, WFL, B}(send, recv)

# X-buffers WFL: complement of the ADJACENT corner.
west_writes_fold_line(arch) = !northwest_writes_fold_line(arch)
east_writes_fold_line(arch) = !northeast_writes_fold_line(arch)

# TripolarXBuffer send: always pack full buffer (Ny_buf rows)
_fill_west_send_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, 1+Hx:2Hx, 1+Hy:size(buff.send,2)+Hy, :)
_fill_east_send_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:size(buff.send,2)+Hy, :)

# TripolarXBuffer recv FL=false: no fold line, write all Ny_buf rows
_recv_from_west_buffer!(c, buff::TripolarXBuffer{false}, Hx, Hy, Nx, Ny) =
    view(c, 1:Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv
_recv_from_east_buffer!(c, buff::TripolarXBuffer{false}, Hx, Hy, Nx, Ny) =
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv

# TripolarXBuffer recv FL=true, WFL=true: write all Ny_buf rows (fold line included)
_recv_from_west_buffer!(c, buff::TripolarXBuffer{true, true}, Hx, Hy, Nx, Ny) =
    view(c, 1:Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv
_recv_from_east_buffer!(c, buff::TripolarXBuffer{true, true}, Hx, Hy, Nx, Ny) =
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv

# TripolarXBuffer recv FL=true, WFL=false: skip last row (the fold line)
_recv_from_west_buffer!(c, buff::TripolarXBuffer{true, false}, Hx, Hy, Nx, Ny) =
    view(c, 1:Hx, 1+Hy:size(buff.recv,2)-1+Hy, :) .= view(buff.recv, :, 1:size(buff.recv,2)-1, :)
_recv_from_east_buffer!(c, buff::TripolarXBuffer{true, false}, Hx, Hy, Nx, Ny) =
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:size(buff.recv,2)-1+Hy, :) .= view(buff.recv, :, 1:size(buff.recv,2)-1, :)

# Fold-aware x-buffer constructors.
# Separate west/east constructors since they complement different corners.
function west_tripolar_buffer(arch, grid, data, Hx, bc, loc,
                              north::TwoDZipperBuffer{<:Any, <:Any, TY}) where TY
    ℓy = north.loc[2]
    topo = fold_topo(north)
    fl = has_fold_line(TY, ℓy)
    Ny_buf = length(ℓy, topo, size(grid, 2))
    wfl = fl && west_writes_fold_line(arch)
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    return TripolarXBuffer{fl, wfl}(send, recv)
end

function east_tripolar_buffer(arch, grid, data, Hx, bc, loc,
                              north::TwoDZipperBuffer{<:Any, <:Any, TY}) where TY
    ℓy = north.loc[2]
    topo = fold_topo(north)
    fl = has_fold_line(TY, ℓy)
    Ny_buf = length(ℓy, topo, size(grid, 2))
    wfl = fl && east_writes_fold_line(arch)
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    return TripolarXBuffer{fl, wfl}(send, recv)
end

# Fallback when north is not a zipper (south ranks)
west_tripolar_buffer(arch, grid, data, Hx, bc, loc, north) = x_communication_buffer(arch, grid, data, Hx, bc)
east_tripolar_buffer(arch, grid, data, Hx, bc, loc, north) = x_communication_buffer(arch, grid, data, Hx, bc)

#####
##### Buffer construction for tripolar grids
#####

function communication_buffers(grid::MPITripolarGridOfSomeKind, data, bcs, loc)
    Hx, Hy, Hz = halo_size(grid)
    arch = architecture(grid)

    south = y_communication_buffer(arch, grid, data, Hy, bcs.south)
    north = y_tripolar_buffer(arch, grid, data, Hy, bcs.north, loc)

    # x-buffers: separate west/east since they complement different corners (NW/NE)
    west  = west_tripolar_buffer(arch, grid, data, Hx, bcs.west, loc, north)
    east  = east_tripolar_buffer(arch, grid, data, Hx, bcs.east, loc, north)


    sw = corner_communication_buffer(arch, grid, data, Hx, Hy, west, south)
    se = corner_communication_buffer(arch, grid, data, Hx, Hy, east, south)
    nw = northwest_tripolar_buffer(arch, grid, data, Hx, Hy, west, north)
    ne = northeast_tripolar_buffer(arch, grid, data, Hx, Hy, east, north)

    return CommunicationBuffers(west, east, south, north, sw, se, nw, ne)
end

# Fallback: non-zipper north BC uses standard buffer
y_tripolar_buffer(arch, grid, data, Hy, bc, loc) = y_communication_buffer(arch, grid, data, Hy, bc)

# 2D fold (MxN) → TwoDZipperBuffer (interior-width, Hy′ = Hy or Hy+1 rows for fold line)
function y_tripolar_buffer(arch, grid::AbstractGrid{<:Any, <:Any, TY},
                           data, Hy, bc::DistributedZipper, loc::Loc) where {TY <: PencilFoldedTopology, Loc}
    Nx = size(grid, 1)
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    sgn = bc.condition.sign
    fl = has_fold_line(TY, loc[2])
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && north_writes_fold_line(arch)
    send = on_architecture(arch, zeros(FT, Nx, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Nx, Hy′, Tz))
    return TwoDZipperBuffer{fl, wfl, TY}(loc, send, recv, sgn)
end

# Fallbacks: non-zipper corners
northwest_tripolar_buffer(arch, grid, data, Hx, Hy, xedge, yedge) = corner_communication_buffer(arch, grid, data, Hx, Hy, xedge, yedge)
northeast_tripolar_buffer(arch, grid, data, Hx, Hy, xedge, yedge) = corner_communication_buffer(arch, grid, data, Hx, Hy, xedge, yedge)

# Corner buffers are only needed for 2D (MxN) partitions.
# FL (fold line in buffer) is true for ALL corners when has_fold_line, to ensure
# MPI size matching between mirror partners. WFL (writes fold line) is per-corner,
# based on whether the corner's global x-position is past the pivot.

function northwest_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{<:Any, <:Any, TY}) where TY
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    ℓx, ℓy = yedge.loc[1], yedge.loc[2]
    fl = has_fold_line(TY, ℓy)
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northwest_writes_fold_line(arch)
    Hx′ = northwest_x_size(ℓx, Hx)
    send = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
    return ZipperCornerBuffer{fl, wfl, TY}(yedge.loc, send, recv, sgn)
end

function northeast_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{<:Any, <:Any, TY}) where TY
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    ℓx, ℓy = yedge.loc[1], yedge.loc[2]
    fl = has_fold_line(TY, ℓy)
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northeast_writes_fold_line(arch)
    Hx′ = northeast_x_size(ℓx, Hx)
    send = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
    return ZipperCornerBuffer{fl, wfl, TY}(yedge.loc, send, recv, sgn)
end

#####
##### Location aliases for dispatch
#####

const CC = Tuple{<:Center, <:Center, <:Any}
const FC = Tuple{<:Face,   <:Center, <:Any}
const CF = Tuple{<:Center, <:Face,   <:Any}
const FF = Tuple{<:Face,   <:Face,   <:Any}

#####
##### Helper functions: compute ranges from location and topology
#####

# Field accessors
@inline fold_topo(::TwoDZipperBuffer{<:Any, <:Any, TY}) where TY = TY()
@inline fold_topo(::ZipperCornerBuffer{<:Any, <:Any, TY}) where TY = TY()

# North buffer: fold-line y-index (source/destination row at the fold)
@inline north_foldline_send_y_index(::UPivotTopology, ::Center, Hy, Ny) = Ny + Hy
@inline north_foldline_send_y_index(::FPivotTopology, ::Face,   Hy, Ny) = Ny + 1 + Hy

@inline north_foldline_recv_y_index(::UPivotTopology, ::Center, Hy, Ny) = Ny + Hy
@inline north_foldline_recv_y_index(::FPivotTopology, ::Face,   Hy, Ny) = Ny + 1 + Hy

# North buffer: halo y-range (reversed for send, forward for recv)
@inline north_halo_send_y_range(::UPivotTopology, ::Center, Hy, Ny) = Ny+Hy-1:-1:Ny
@inline north_halo_send_y_range(topo,             loc_y,    Hy, Ny) = Ny+Hy:-1:Ny+1

@inline north_halo_recv_y_range(::FPivotTopology, ::Face, Hy, Ny) = 2+Ny+Hy:1+Ny+2Hy
@inline north_halo_recv_y_range(topo,             loc_y,  Hy, Ny) = 1+Ny+Hy:Ny+2Hy

# North buffer: recv x-range
@inline north_recv_x_range(::Center, Hx, Nx) = 1+Hx:Nx+Hx
@inline north_recv_x_range(::Face,   Hx, Nx) = 2+Hx:Nx+Hx+1

# Corner buffers: send x-ranges (reversed for fold)
@inline northwest_send_x_range(::Center, Hx, Nx) = 2Hx:-1:1+Hx
@inline northwest_send_x_range(::Face,   Hx, Nx) = 2Hx+1:-1:1+Hx
@inline northeast_send_x_range(::Center, Hx, Nx) = Nx+Hx:-1:1+Nx
@inline northeast_send_x_range(::Face,   Hx, Nx) = Nx+Hx:-1:Nx+2

# Corner buffers: recv x-ranges
@inline northwest_recv_x_range(::Center, Hx, Nx) = 1:Hx
@inline northwest_recv_x_range(::Face,   Hx, Nx) = 1:Hx+1
@inline northeast_recv_x_range(::Center, Hx, Nx) = 1+Nx+Hx:Nx+2Hx
@inline northeast_recv_x_range(::Face,   Hx, Nx) = 2+Nx+Hx:Nx+2Hx

# Corner buffers: x-size (for constructors)
@inline northwest_x_size(::Center, Hx) = Hx
@inline northwest_x_size(::Face,   Hx) = Hx + 1
@inline northeast_x_size(::Center, Hx) = Hx
@inline northeast_x_size(::Face,   Hx) = Hx - 1

#####
##### TwoDZipperBuffer: north send (FL=true has fold line, FL=false does not)
#####

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{true}, Hx, Hy, Nx, Ny)
    topo, ℓy = fold_topo(b), b.loc[2]
    j = north_foldline_send_y_index(topo, ℓy, Hy, Ny)
    jrange = north_halo_send_y_range(topo, ℓy, Hy, Ny)
    view(b.send, :, 1:1, :)    .= b.sign .* view(c, Nx+Hx:-1:1+Hx, j:j, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, jrange, :)
end

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{false}, Hx, Hy, Nx, Ny)
    topo, ℓy = fold_topo(b), b.loc[2]
    jrange = north_halo_send_y_range(topo, ℓy, Hy, Ny)
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, jrange, :)
end

#####
##### TwoDZipperBuffer: north recv (FL × WFL dispatch)
#####

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{true, true}, Hx, Hy, Nx, Ny)
    topo, ℓy = fold_topo(buff), buff.loc[2]
    j = north_foldline_recv_y_index(topo, ℓy, Hy, Ny)
    xr = north_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(topo, ℓy, Hy, Ny)
    view(c, xr, j:j   , :) .= view(buff.recv, :, 1:1   , :)
    view(c, xr, yr, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{true, false}, Hx, Hy, Nx, Ny)
    xr = north_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(fold_topo(buff), buff.loc[2], Hy, Ny)
    view(c, xr, yr, :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{false}, Hx, Hy, Nx, Ny)
    xr = north_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(fold_topo(buff), buff.loc[2], Hy, Ny)
    view(c, xr, yr, :) .= buff.recv
end

#####
##### Corner send: NW and NE (FL=true includes fold line, FL=false halo only)
#####

function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{true}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    ℓy = b.loc[2]
    j = north_foldline_send_y_index(topo, ℓy, Hy, Ny)
    xr = northwest_send_x_range(b.loc[1], Hx, Nx)
    b.send .= b.sign .* view(c, xr, j:-1:j-Hy, :)
end

function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    yr = north_halo_send_y_range(topo, b.loc[2], Hy, Ny)
    xr = northwest_send_x_range(b.loc[1], Hx, Nx)
    b.send .= b.sign .* view(c, xr, yr, :)
end

function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{true}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    ℓy = b.loc[2]
    j = north_foldline_send_y_index(topo, ℓy, Hy, Ny)
    xr = northeast_send_x_range(b.loc[1], Hx, Nx)
    b.send .= b.sign .* view(c, xr, j:-1:j-Hy, :)
end

function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    yr = north_halo_send_y_range(topo, b.loc[2], Hy, Ny)
    xr = northeast_send_x_range(b.loc[1], Hx, Nx)
    b.send .= b.sign .* view(c, xr, yr, :)
end

#####
##### Corner recv: NW and NE (FL × WFL dispatch)
#####

function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{true, true}, Hx, Hy, Nx, Ny)
    topo, ℓy = fold_topo(buff), buff.loc[2]
    j = north_foldline_recv_y_index(topo, ℓy, Hy, Ny)
    xr = northwest_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(topo, ℓy, Hy, Ny)
    view(c, xr, j:j, :) .= view(buff.recv, :, 1:1, :)
    view(c, xr,  yr, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{true, false}, Hx, Hy, Nx, Ny)
    xr = northwest_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(fold_topo(buff), buff.loc[2], Hy, Ny)
    view(c, xr, yr, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny)
    xr = northwest_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(fold_topo(buff), buff.loc[2], Hy, Ny)
    view(c, xr, yr, :) .= buff.recv
end

function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{true, true}, Hx, Hy, Nx, Ny)
    topo, ℓy = fold_topo(buff), buff.loc[2]
    xr = northeast_recv_x_range(buff.loc[1], Hx, Nx)
    j = north_foldline_recv_y_index(topo, ℓy, Hy, Ny)
    yr = north_halo_recv_y_range(topo, ℓy, Hy, Ny)
    view(c, xr, j:j, :) .= view(buff.recv, :, 1:1, :)
    view(c, xr, yr , :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{true, false}, Hx, Hy, Nx, Ny)
    xr = northeast_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(fold_topo(buff), buff.loc[2], Hy, Ny)
    view(c, xr, yr, :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny)
    xr = northeast_recv_x_range(buff.loc[1], Hx, Nx)
    yr = north_halo_recv_y_range(fold_topo(buff), buff.loc[2], Hy, Ny)
    view(c, xr, yr, :) .= buff.recv
end
