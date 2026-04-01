using Oceananigans.BoundaryConditions: get_boundary_kernels, DistributedCommunication
using Oceananigans.DistributedComputations: cooperative_waitall!, recv_from_buffers!, distributed_fill_halo_event!
using Oceananigans.DistributedComputations: CommunicationBuffers, fill_corners!, loc_id, AsynchronousDistributed
using Oceananigans.Grids: AbstractGrid, topology,
    RightCenterFolded, RightFaceFolded,
    LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
    LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected
using Oceananigans.DistributedComputations: Distributed, on_architecture, ranks, x_communication_buffer

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

struct TwoDZipperBuffer{FL, WFL, Loc, FoT, B, S}
    send :: B
    recv :: B
    sign :: S
end

struct ZipperCornerBuffer{FL, WFL, Loc, FoT, B, S}
    send :: B
    recv :: B
    sign :: S
end

# Value-argument constructors: all types inferred
OneDZipperBuffer(loc::Loc, fot::FoT, send::B, recv::B, sign::S) where {Loc, FoT, B, S} = OneDZipperBuffer{Loc, FoT, B, S}(send, recv, sign)
TwoDZipperBuffer(loc, fot, send, recv, sign, ::Val{FL}, ::Val{WFL}) where {FL, WFL} =
    TwoDZipperBuffer{FL, WFL, typeof(loc), typeof(fot), typeof(send), typeof(sign)}(send, recv, sign)
ZipperCornerBuffer(loc, fot, send, recv, sign, ::Val{FL}, ::Val{WFL}) where {FL, WFL} =
    ZipperCornerBuffer{FL, WFL, typeof(loc), typeof(fot), typeof(send), typeof(sign)}(send, recv, sign)
ZipperCornerBuffer(::Type{Loc}, ::Type{FoT}, send, recv, sign, ::Val{FL}, ::Val{WFL}) where {Loc, FoT, FL, WFL} =
    ZipperCornerBuffer{FL, WFL, Loc, FoT, typeof(send), typeof(sign)}(send, recv, sign)

Adapt.adapt_structure(to, buff::OneDZipperBuffer) = nothing
Adapt.adapt_structure(to, buff::TwoDZipperBuffer) = nothing
Adapt.adapt_structure(to, buff::ZipperCornerBuffer) = nothing

# X-direction buffer for tripolar grids: like TwoDBuffer but with location-aware y-size
# and fold-line awareness. FL/WFL mirror the corner buffer pattern:
# - FL: buffer has fold-line row (same as corners, for MPI size matching)
# - WFL: recv writes the fold-line row (complement of the adjacent corner)
#   West WFL = !NW corner WFL, East WFL = !NE corner WFL.
struct TripolarXBuffer{B, FL, WFL}
    send :: B
    recv :: B
end

Adapt.adapt_structure(to, buff::TripolarXBuffer) = nothing

TripolarXBuffer(send, recv, ::Val{FL}, ::Val{WFL}) where {FL, WFL} =
    TripolarXBuffer{typeof(send), FL, WFL}(send, recv)

# X-buffers WFL: complement of the ADJACENT corner.
west_writes_fold_line(arch) = !northwest_writes_fold_line(arch)
east_writes_fold_line(arch) = !northeast_writes_fold_line(arch)

# TripolarXBuffer send: always pack full buffer (Ny_buf rows)
_fill_west_send_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, 1+Hx:2Hx, 1+Hy:size(buff.send,2)+Hy, :)
_fill_east_send_buffer!(c, buff::TripolarXBuffer, Hx, Hy, Nx, Ny) =
    buff.send .= view(c, 1+Nx:Nx+Hx, 1+Hy:size(buff.send,2)+Hy, :)

# TripolarXBuffer recv FL=false: no fold line, write all Ny_buf rows
_recv_from_west_buffer!(c, buff::TripolarXBuffer{<:Any, false}, Hx, Hy, Nx, Ny) =
    view(c, 1:Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv
_recv_from_east_buffer!(c, buff::TripolarXBuffer{<:Any, false}, Hx, Hy, Nx, Ny) =
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv

# TripolarXBuffer recv FL=true, WFL=true: write all Ny_buf rows (fold line included)
_recv_from_west_buffer!(c, buff::TripolarXBuffer{<:Any, true, true}, Hx, Hy, Nx, Ny) =
    view(c, 1:Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv
_recv_from_east_buffer!(c, buff::TripolarXBuffer{<:Any, true, true}, Hx, Hy, Nx, Ny) =
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:size(buff.recv,2)+Hy, :) .= buff.recv

# TripolarXBuffer recv FL=true, WFL=false: skip last row (the fold line)
_recv_from_west_buffer!(c, buff::TripolarXBuffer{<:Any, true, false}, Hx, Hy, Nx, Ny) =
    view(c, 1:Hx, 1+Hy:size(buff.recv,2)-1+Hy, :) .= view(buff.recv, :, 1:size(buff.recv,2)-1, :)
_recv_from_east_buffer!(c, buff::TripolarXBuffer{<:Any, true, false}, Hx, Hy, Nx, Ny) =
    view(c, 1+Nx+Hx:Nx+2Hx, 1+Hy:size(buff.recv,2)-1+Hy, :) .= view(buff.recv, :, 1:size(buff.recv,2)-1, :)

# Fold-aware x-buffer constructors.
# Fallback for non-zipper north (south ranks): standard x-communication.
# Separate west/east constructors since they complement different corners.
function west_tripolar_buffer(arch, grid, data, Hx, bc, loc,
                              north::TwoDZipperBuffer{<:Any, <:Any, Loc, FoT}) where {Loc, FoT}
    loc_y = Loc.parameters[2]()
    fl = has_fold_line(FoT, loc_y)
    Ny_buf = length(loc_y, FoT(), size(grid, 2))
    wfl = fl && west_writes_fold_line(arch)
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    return TripolarXBuffer(send, recv, Val(fl), Val(wfl))
end

function east_tripolar_buffer(arch, grid, data, Hx, bc, loc,
                              north::TwoDZipperBuffer{<:Any, <:Any, Loc, FoT}) where {Loc, FoT}
    loc_y = Loc.parameters[2]()
    fl = has_fold_line(FoT, loc_y)
    Ny_buf = length(loc_y, FoT(), size(grid, 2))
    wfl = fl && east_writes_fold_line(arch)
    _, _, Tz = size(parent(data))
    FT = eltype(data)
    send = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    recv = on_architecture(arch, zeros(FT, Hx, Ny_buf, Tz))
    return TripolarXBuffer(send, recv, Val(fl), Val(wfl))
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

function northwest_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{<:Any, <:Any, Loc, FoT}) where {Loc, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    loc_x_inst = instantiate(Loc.parameters[1])
    fl = has_fold_line(FoT, Loc.parameters[2]())
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northwest_writes_fold_line(arch)
    Hx′ = nw_corner_nx(loc_x_inst, Hx)
    send = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
    return ZipperCornerBuffer(Loc, FoT, send, recv, sgn, Val(fl), Val(wfl))
end

function northeast_tripolar_buffer(arch, grid, data, Hx, Hy, xedge,
                                   yedge::TwoDZipperBuffer{<:Any, <:Any, Loc, FoT}) where {Loc, FoT}
    Tz = size(parent(data), 3); FT = eltype(data); sgn = yedge.sign
    loc_x_inst = instantiate(Loc.parameters[1])
    fl = has_fold_line(FoT, Loc.parameters[2]())
    Hy′ = fl ? Hy + 1 : Hy
    wfl = fl && northeast_writes_fold_line(arch)
    Hx′ = ne_corner_nx(loc_x_inst, Hx)
    send = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
    recv = on_architecture(arch, zeros(FT, Hx′, Hy′, Tz))
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
##### Helper functions: compute ranges from location and topology
#####

# Type-parameter accessors
@inline loc_x(::TwoDZipperBuffer{<:Any, <:Any, Loc}) where Loc = instantiate(Loc.parameters[1])
@inline loc_y(::TwoDZipperBuffer{<:Any, <:Any, Loc}) where Loc = instantiate(Loc.parameters[2])
@inline fold_topo(::TwoDZipperBuffer{<:Any, <:Any, <:Any, FoT}) where FoT = FoT()

@inline loc_x(::ZipperCornerBuffer{<:Any, <:Any, Loc}) where Loc = instantiate(Loc.parameters[1])
@inline loc_y(::ZipperCornerBuffer{<:Any, <:Any, Loc}) where Loc = instantiate(Loc.parameters[2])
@inline fold_topo(::ZipperCornerBuffer{<:Any, <:Any, <:Any, FoT}) where FoT = FoT()

@inline loc_x(::OneDZipperBuffer{Loc}) where Loc = instantiate(Loc.parameters[1])
@inline loc_y(::OneDZipperBuffer{Loc}) where Loc = instantiate(Loc.parameters[2])
@inline fold_topo(::OneDZipperBuffer{<:Any, FoT}) where FoT = FoT()

# Send y-ranges (source rows, reversed for fold)
@inline send_fold_y(::UPivotTopology, ::Center, Hy, Ny) = Ny + Hy
@inline send_fold_y(::FPivotTopology, ::Face,   Hy, Ny) = Ny + 1 + Hy

@inline send_halo_y(::UPivotTopology, ::Center, Hy, Ny) = Ny+Hy-1:-1:Ny
@inline send_halo_y(topo,             loc_y,    Hy, Ny) = Ny+Hy:-1:Ny+1

# Recv y-ranges (destination rows in parent)
@inline recv_fold_y(::UPivotTopology, ::Center, Hy, Ny) = Ny + Hy
@inline recv_fold_y(::FPivotTopology, ::Face,   Hy, Ny) = Ny + 1 + Hy

@inline recv_halo_y(::FPivotTopology, ::Face, Hy, Ny) = 2+Ny+Hy:1+Ny+2Hy
@inline recv_halo_y(topo,             loc_y,  Hy, Ny) = 1+Ny+Hy:Ny+2Hy

# Recv x-range for north buffer
@inline recv_x_range(::Center, Hx, Nx) = 1+Hx:Nx+Hx
@inline recv_x_range(::Face,   Hx, Nx) = 2+Hx:Nx+Hx+1

# Corner send x-ranges (reversed for fold)
@inline nw_send_x(::Center, Hx, Nx) = 2Hx:-1:1+Hx
@inline nw_send_x(::Face,   Hx, Nx) = 2Hx+1:-1:1+Hx
@inline ne_send_x(::Center, Hx, Nx) = Nx+Hx:-1:1+Nx
@inline ne_send_x(::Face,   Hx, Nx) = Nx+Hx:-1:Nx+2

# Corner recv x-ranges
@inline nw_recv_x(::Center, Hx, Nx) = 1:Hx
@inline nw_recv_x(::Face,   Hx, Nx) = 1:Hx+1
@inline ne_recv_x(::Center, Hx, Nx) = 1+Nx+Hx:Nx+2Hx
@inline ne_recv_x(::Face,   Hx, Nx) = 2+Nx+Hx:Nx+2Hx

# Corner buffer x-size (for constructors)
@inline nw_corner_nx(::Center, Hx) = Hx
@inline nw_corner_nx(::Face,   Hx) = Hx + 1
@inline ne_corner_nx(::Center, Hx) = Hx
@inline ne_corner_nx(::Face,   Hx) = Hx - 1

#####
##### TwoDZipperBuffer: north send (FL=true has fold line, FL=false does not)
#####

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{true}, Hx, Hy, Nx, Ny)
    topo, ly = fold_topo(b), loc_y(b)
    fy = send_fold_y(topo, ly, Hy, Ny)
    view(b.send, :, 1:1, :)    .= b.sign .* view(c, Nx+Hx:-1:1+Hx, fy:fy, :)
    view(b.send, :, 2:Hy+1, :) .= b.sign .* view(c, Nx+Hx:-1:1+Hx, send_halo_y(topo, ly, Hy, Ny), :)
end

function _fill_north_send_buffer!(c, b::TwoDZipperBuffer{false}, Hx, Hy, Nx, Ny)
    topo, ly = fold_topo(b), loc_y(b)
    b.send .= b.sign .* view(c, Nx+Hx:-1:1+Hx, send_halo_y(topo, ly, Hy, Ny), :)
end

#####
##### TwoDZipperBuffer: north recv (FL × WFL dispatch)
#####

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{true, true}, Hx, Hy, Nx, Ny)
    xr = recv_x_range(loc_x(buff), Hx, Nx)
    topo, ly = fold_topo(buff), loc_y(buff)
    fy = recv_fold_y(topo, ly, Hy, Ny)
    view(c, xr, fy:fy                        , :) .= view(buff.recv, :, 1:1   , :)
    view(c, xr, recv_halo_y(topo, ly, Hy, Ny), :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{true, false}, Hx, Hy, Nx, Ny)
    xr = recv_x_range(loc_x(buff), Hx, Nx)
    view(c, xr, recv_halo_y(fold_topo(buff), loc_y(buff), Hy, Ny), :) .= view(buff.recv, :, 2:Hy+1, :)
end

function _recv_from_north_buffer!(c, buff::TwoDZipperBuffer{false}, Hx, Hy, Nx, Ny)
    xr = recv_x_range(loc_x(buff), Hx, Nx)
    view(c, xr, recv_halo_y(fold_topo(buff), loc_y(buff), Hy, Ny), :) .= buff.recv
end

#####
##### OneDZipperBuffer: north send (Center-x full reverse, Face-x partial reverse)
#####

function _fill_north_send_buffer!(c, b::OneDZipperBuffer{<:Union{CC,CF}}, Hx, Hy, Nx, Ny)
    b.send .= b.sign .* view(c, size(c,1):-1:1, send_halo_y(fold_topo(b), loc_y(b), Hy, Ny), :)
end

function _fill_north_send_buffer!(c, b::OneDZipperBuffer{<:Union{FC,FF}}, Hx, Hy, Nx, Ny)
    Tx = size(c, 1)
    hy = send_halo_y(fold_topo(b), loc_y(b), Hy, Ny)
    view(b.send, 2:Tx, :, :) .= b.sign .* view(c, Tx:-1:2, hy, :)
    view(b.send, 1:1,  :, :) .= b.sign .* view(c,     1:1, hy, :)
end

#####
##### OneDZipperBuffer: north recv
#####

_recv_from_north_buffer!(c, buff::OneDZipperBuffer, Hx, Hy, Nx, Ny) = view(c, :, recv_halo_y(fold_topo(buff), loc_y(buff), Hy, Ny), :) .= buff.recv

#####
##### Corner send: NW and NE (FL=true includes fold line, FL=false halo only)
#####

function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{true}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    ly = loc_y(b)
    fy = send_fold_y(topo, ly, Hy, Ny)
    b.send .= b.sign .* view(c, nw_send_x(loc_x(b), Hx, Nx), fy:-1:fy-Hy, :)
end

function _fill_northwest_send_buffer!(c, b::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    fy = send_halo_y(topo, loc_y(b), Hy, Ny)
    b.send .= b.sign .* view(c, nw_send_x(loc_x(b), Hx, Nx), fy, :)
end

function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{true}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    ly = loc_y(b)
    fy = send_fold_y(topo, ly, Hy, Ny)
    b.send .= b.sign .* view(c, ne_send_x(loc_x(b), Hx, Nx), fy:-1:fy-Hy, :)
end

function _fill_northeast_send_buffer!(c, b::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny)
    topo = fold_topo(b)
    fy = send_halo_y(topo, loc_y(b), Hy, Ny)
    b.send .= b.sign .* view(c, ne_send_x(loc_x(b), Hx, Nx), fy, :)
end

#####
##### Corner recv: NW and NE (FL × WFL dispatch)
#####

function _recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{true, true}, Hx, Hy, Nx, Ny)
    xr = nw_recv_x(loc_x(buff), Hx, Nx)
    topo, ly = fold_topo(buff), loc_y(buff)
    fy = recv_fold_y(topo, ly, Hy, Ny)
    view(c, xr, fy:fy, :)                       .= view(buff.recv, :, 1:1, :)
    view(c, xr, recv_halo_y(topo, ly, Hy, Ny), :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{true, false}, Hx, Hy, Nx, Ny) =
    view(c, nw_recv_x(loc_x(buff), Hx, Nx), recv_halo_y(fold_topo(buff), loc_y(buff), Hy, Ny), :) .=
        view(buff.recv, :, 2:size(buff.recv,2), :)

_recv_from_northwest_buffer!(c, buff::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny) =
    view(c, nw_recv_x(loc_x(buff), Hx, Nx), recv_halo_y(fold_topo(buff), loc_y(buff), Hy, Ny), :) .= buff.recv

function _recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{true, true}, Hx, Hy, Nx, Ny)
    xr = ne_recv_x(loc_x(buff), Hx, Nx)
    topo, ly = fold_topo(buff), loc_y(buff)
    fy = recv_fold_y(topo, ly, Hy, Ny)
    view(c, xr, fy:fy, :)                       .= view(buff.recv, :, 1:1, :)
    view(c, xr, recv_halo_y(topo, ly, Hy, Ny), :) .= view(buff.recv, :, 2:size(buff.recv,2), :)
end

_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{true, false}, Hx, Hy, Nx, Ny) =
    view(c, ne_recv_x(loc_x(buff), Hx, Nx), recv_halo_y(fold_topo(buff), loc_y(buff), Hy, Ny), :) .=
        view(buff.recv, :, 2:size(buff.recv,2), :)

_recv_from_northeast_buffer!(c, buff::ZipperCornerBuffer{false}, Hx, Hy, Nx, Ny) =
    view(c, ne_recv_x(loc_x(buff), Hx, Nx), recv_halo_y(fold_topo(buff), loc_y(buff), Hy, Ny), :) .= buff.recv

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
