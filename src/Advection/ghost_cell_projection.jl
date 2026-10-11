using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField, interior
using Oceananigans.Grids: topology
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Axᶠᶜᶜ, Ayᶜᶠᶜ, Azᶜᶜᶠ, Vᶜᶜᶜ
using Oceananigans.Utils: KernelParameters, launch!
using KernelAbstractions: @index, @kernel

#####
##### Cell-wise projection of the tracer values reconstructed in the cells whose stencils read ghost values
#####
#
# For a cell with inflow Courant number ν, inflow mean cⁱⁿ and outflow mean cᵒᵘᵗ, a forward-Euler update reads
#     cⁿ⁺¹ = c + ν (cⁱⁿ - c) - ν (cᵒᵘᵗ - c).
# If every reconstructed value lies within the range of the cell it flows into and of that cell's neighbors, and
#     c - (1 - ν)/ν (cᵐᵃˣ - c) ≤ cᵒᵘᵗ ≤ c + (1 - ν)/ν (c - cᵐⁱⁿ),
# with cᵐⁱⁿ, cᵐᵃˣ the range of the cell and its neighbors, then cᵐⁱⁿ ≤ cⁿ⁺¹ ≤ cᵐᵃˣ in any dimension.
# The outflow values closest to the WENO ones that satisfy these conditions are clamp(ĉ - λ, I), with one shift λ per cell.

const BoundedGhostCellWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells{<:Any, <:NamedTuple}}

#####
##### Static mask: bits 1-6 flag the active neighbors (west, east, south, north, bottom, top) and bit 7 flags
##### the band of active cells within `N` cells of an inactive cell along a grid line
#####

@inline active_center(i, j, k, grid) = active_node(i, j, k, grid, Center(), Center(), Center())

@inline function inactive_within(i, j, k, grid, e, ::Val{N}) where N
    offsets = ntuple(o -> ifelse(o ≤ N, o - N - 1, o - N), Val(2N))
    return mapreduce(o -> !active_center(((i, j, k) .+ o .* e)..., grid), |, offsets)
end

@inline near_inactive_x(i, j, k, grid, n) = inactive_within(i, j, k, grid, (1, 0, 0), n)
@inline near_inactive_y(i, j, k, grid, n) = inactive_within(i, j, k, grid, (0, 1, 0), n)
@inline near_inactive_z(i, j, k, grid, n) = inactive_within(i, j, k, grid, (0, 0, 1), n)
@inline near_inactive_x(i, j, k, ::XFG, n) = false
@inline near_inactive_y(i, j, k, ::YFG, n) = false
@inline near_inactive_z(i, j, k, ::ZFG, n) = false

@inline active_neighbors_x(i, j, k, grid) = (active_center(i - 1, j, k, grid), active_center(i + 1, j, k, grid))
@inline active_neighbors_y(i, j, k, grid) = (active_center(i, j - 1, k, grid), active_center(i, j + 1, k, grid))
@inline active_neighbors_z(i, j, k, grid) = (active_center(i, j, k - 1, grid), active_center(i, j, k + 1, grid))
@inline active_neighbors_x(i, j, k, ::XFG) = (false, false)
@inline active_neighbors_y(i, j, k, ::YFG) = (false, false)
@inline active_neighbors_z(i, j, k, ::ZFG) = (false, false)

@kernel function _compute_ghost_cell_mask!(mask, grid, n)
    i, j, k = @index(Global, NTuple)
    near = near_inactive_x(i, j, k, grid, n) | near_inactive_y(i, j, k, grid, n) | near_inactive_z(i, j, k, grid, n)
    flags = (active_neighbors_x(i, j, k, grid)...,
             active_neighbors_y(i, j, k, grid)...,
             active_neighbors_z(i, j, k, grid)...,
             near & active_center(i, j, k, grid))
    bits = ntuple(b -> UInt8(flags[b]) << (b - 1), Val(7))
    @inbounds mask[i, j, k] = reduce(|, bits)
end

@inline in_band(mask, i, j, k) = @inbounds (mask[i, j, k] >> 6) & 0x01 == 0x01
@inline active_neighbor(m, b) = (m >> (b - 1)) & 0x01 == 0x01

"""
    ghost_cell_projection(grid, N)

Allocate the state of the projection of a bounded `GhostCells` scheme of a `WENO` with `N` stencils: the neighbor mask,
the list of band cells, the shift, and the face values reconstructed by `update_ghost_cell_projection!`.
The state is a workspace that tracers advected with schemes of the same order share, since every tracer
updates it right before its own tendency is computed.
"""
function ghost_cell_projection(grid, N)
    arch = architecture(grid)
    mask = CenterField(grid, UInt8)
    launch!(arch, grid, :xyz, _compute_ghost_cell_mask!, mask, grid, Val(N))
    fill_halo_regions!(mask)

    # The shift kernel is mapped over the band cells, so that all its threads do the same work
    band_indices = findall(m -> (m >> 6) & 0x01 == 0x01, on_architecture(CPU(), interior(mask)))
    IntType = minimal_index_type(maximum(size(grid)))
    band = on_architecture(arch, map(I -> IntType.(Tuple(I)), band_indices))

    shift = CenterField(grid)
    faces = (x = XFaceField(grid), y = YFaceField(grid), z = ZFaceField(grid))

    return (; band, mask, shift, faces)
end

minimal_index_type(N) = N ≤ typemax(UInt8) ? UInt8 : N ≤ typemax(UInt16) ? UInt16 : N ≤ typemax(UInt32) ? UInt32 : UInt64

"""
    materialize_ghost_cell_projection(scheme, workspaces, grid)

Return `scheme` with the projection state of its bounded `GhostCells` boundary scheme, taken from `workspaces`
(a dictionary keyed by the number of WENO stencils) or allocated and added to it.
"""
materialize_ghost_cell_projection(scheme, workspaces, grid) = scheme

function materialize_ghost_cell_projection(weno::WENO{N, FT, WCT}, workspaces, grid) where {N, FT, WCT}
    buffer = weno.buffer_scheme
    buffer isa GhostCells && buffer.bounded || return weno
    projection = get!(() -> ghost_cell_projection(grid, N), workspaces, N)
    bounded_buffer = GhostCells(buffer.scheme, projection, true)
    return WENO{N, FT, WCT}(weno.bounds, bounded_buffer, weno.advecting_velocity_scheme, weno.time_discretization)
end

# The projection state belongs to the tracer that last updated it, so the fluxes of other fields use the plain ghost cells
@inline without_ghost_cell_projection(scheme) = scheme

@inline function without_ghost_cell_projection(weno::WENO{N, FT, WCT, <:Any, <:Any, <:GhostCells{<:Any, <:NamedTuple}}) where {N, FT, WCT}
    buffer = weno.buffer_scheme
    return WENO{N, FT, WCT}(weno.bounds, GhostCells(buffer.scheme, nothing, false), weno.advecting_velocity_scheme, weno.time_discretization)
end

#####
##### Range of a cell and its active neighbors
#####

@inline neighbor_or_self(m, b, c, cᵢ, i, j, k) = ifelse(active_neighbor(m, b), @inbounds(c[i, j, k]), cᵢ)

@inline function neighborhood_range_x(i, j, k, grid, c, cᵢ, m)
    w = neighbor_or_self(m, 1, c, cᵢ, i - 1, j, k)
    e = neighbor_or_self(m, 2, c, cᵢ, i + 1, j, k)
    return min(w, e), max(w, e)
end

@inline function neighborhood_range_y(i, j, k, grid, c, cᵢ, m)
    s = neighbor_or_self(m, 3, c, cᵢ, i, j - 1, k)
    n = neighbor_or_self(m, 4, c, cᵢ, i, j + 1, k)
    return min(s, n), max(s, n)
end

@inline function neighborhood_range_z(i, j, k, grid, c, cᵢ, m)
    b = neighbor_or_self(m, 5, c, cᵢ, i, j, k - 1)
    t = neighbor_or_self(m, 6, c, cᵢ, i, j, k + 1)
    return min(b, t), max(b, t)
end

@inline neighborhood_range_x(i, j, k, ::XFG, c, cᵢ, m) = (cᵢ, cᵢ)
@inline neighborhood_range_y(i, j, k, ::YFG, c, cᵢ, m) = (cᵢ, cᵢ)
@inline neighborhood_range_z(i, j, k, ::ZFG, c, cᵢ, m) = (cᵢ, cᵢ)

@inline function neighborhood_range(i, j, k, grid, c, mask)
    cᵢ = @inbounds c[i, j, k]
    m  = @inbounds mask[i, j, k]
    xᵐⁱⁿ, xᵐᵃˣ = neighborhood_range_x(i, j, k, grid, c, cᵢ, m)
    yᵐⁱⁿ, yᵐᵃˣ = neighborhood_range_y(i, j, k, grid, c, cᵢ, m)
    zᵐⁱⁿ, zᵐᵃˣ = neighborhood_range_z(i, j, k, grid, c, cᵢ, m)
    return min(cᵢ, xᵐⁱⁿ, yᵐⁱⁿ, zᵐⁱⁿ), max(cᵢ, xᵐᵃˣ, yᵐᵃˣ, zᵐᵃˣ)
end

#####
##### Projected tracer fluxes
#####

# `L` and `R` are the cells on the two sides of the face; `ũ` the velocity through it.
@inline function projected_value(ĉ, ũ, L, R, grid, projection, c)
    upwind = ũ > 0
    up   = ifelse(upwind, L, R)
    down = ifelse(upwind, R, L)
    λ = ifelse(in_band(projection.mask, up...), @inbounds(projection.shift[up...]), zero(ĉ))
    cᵐⁱⁿ, cᵐᵃˣ = neighborhood_range(down..., grid, c, projection.mask)
    projected = clamp(clamp(ĉ, cᵐⁱⁿ, cᵐᵃˣ) - λ, cᵐⁱⁿ, cᵐᵃˣ)
    return ifelse(in_band(projection.mask, L...) | in_band(projection.mask, R...), projected, ĉ)
end

# The face values are reconstructed once per stage by `update_ghost_cell_projection!`
for (dir, ξ, area) in ((1, :x, :Axᶠᶜᶜ), (2, :y, :Ayᶜᶠᶜ), (3, :z, :Azᶜᶜᶠ))
    flux = Symbol(:advective_tracer_flux_, ξ)
    FG = (:XFG, :YFG, :ZFG)[dir]
    L = (:((i - 1, j, k)), :((i, j - 1, k)), :((i, j, k - 1)))[dir]

    @eval begin
        @inline function $flux(i, j, k, grid, scheme::BoundedGhostCellWENO, ::ETD, U, c)
            projection = scheme.buffer_scheme.projection
            @inbounds ũ = U[i, j, k]
            @inbounds ĉ = projection.faces.$ξ[i, j, k]
            c̃ = projected_value(ĉ, ũ, $L, (i, j, k), grid, projection, c)
            return $area(i, j, k, grid) * ũ * c̃
        end

        @inline $flux(i, j, k, grid::$FG, ::BoundedGhostCellWENO, ::ETD, U, c) = zero(grid)
    end
end

#####
##### Reconstructed face values
#####

for (dir, ξ, u) in ((1, :x, :u), (2, :y, :v), (3, :z, :w))
    face_value = Symbol(:ghost_cell_face_value_, ξ)
    interpolate = Symbol(:_biased_interpolate_, ξ, ntuple(d -> d == dir ? :ᶠ : :ᵃ, 3)...)
    FG = (:XFG, :YFG, :ZFG)[dir]

    @eval begin
        @inline $face_value(i, j, k, grid, advection, U, c) = @inbounds $interpolate(i, j, k, grid, advection, bias(U.$u[i, j, k]), c)
        @inline $face_value(i, j, k, grid::$FG, advection, U, c) = zero(grid)
    end
end

@kernel function _compute_ghost_cell_face_values!(faces, grid, advection, U, c)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        faces.x[i, j, k] = ghost_cell_face_value_x(i, j, k, grid, advection, U, c)
        faces.y[i, j, k] = ghost_cell_face_value_y(i, j, k, grid, advection, U, c)
        faces.z[i, j, k] = ghost_cell_face_value_z(i, j, k, grid, advection, U, c)
    end
end

@kernel function _compute_ghost_cell_last_face_values!(ĉ, face_value, grid, advection, U, c)
    i, j, k = @index(Global, NTuple)
    @inbounds ĉ[i, j, k] = face_value(i, j, k, grid, advection, U, c)
end

# The faces 1 to N of the interior cells, and then the last face N + 1 of every non-flat direction.
# Launching over the N + 1 faces at once adds an incomplete workgroup in each direction.
function compute_ghost_cell_face_values!(faces, grid, advection, U, c)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, :xyz, _compute_ghost_cell_face_values!, faces, grid, advection, U, c)

    topology(grid, 1) === Flat ||
        launch!(arch, grid, KernelParameters(Nx+1:Nx+1, 1:Ny, 1:Nz), _compute_ghost_cell_last_face_values!, faces.x, ghost_cell_face_value_x, grid, advection, U, c)
    topology(grid, 2) === Flat ||
        launch!(arch, grid, KernelParameters(1:Nx, Ny+1:Ny+1, 1:Nz), _compute_ghost_cell_last_face_values!, faces.y, ghost_cell_face_value_y, grid, advection, U, c)
    topology(grid, 3) === Flat ||
        launch!(arch, grid, KernelParameters(1:Nx, 1:Ny, Nz+1:Nz+1), _compute_ghost_cell_last_face_values!, faces.z, ghost_cell_face_value_z, grid, advection, U, c)

    return nothing
end

#####
##### The shift of one band cell
#####

# Outflow transport, WENO value and admissible interval of the face of cell `(i, j, k)` on side `s`
for (dir, ξ, area, u) in ((1, :x, :Axᶠᶜᶜ, :u), (2, :y, :Ayᶜᶠᶜ, :v), (3, :z, :Azᶜᶜᶠ, :w))
    face_state = Symbol(:face_state_, ξ)
    FG = (:XFG, :YFG, :ZFG)[dir]
    face = Any[:i, :j, :k]
    face[dir] = :(ifelse(s > 0, $(face[dir]) + 1, $(face[dir])))
    neighbor = Any[:i, :j, :k]
    neighbor[dir] = :($(neighbor[dir]) + s)

    @eval begin
        @inline function $face_state(i, j, k, grid, projection, U, c, s)
            f = ($(face...),)
            @inbounds ũ = U.$u[f...]
            # Faces to inactive neighbors carry no flux, whatever the velocity there
            open_face = active_neighbor(@inbounds(projection.mask[i, j, k]), $(2dir - 1) + (s > 0))
            T = ifelse(open_face, s * $area(f..., grid) * ũ, zero(ũ))
            # On a closed face the value and the range are read from inactive cells, and must not reach the sums below
            ĉ = ifelse(open_face, @inbounds(projection.faces.$ξ[f...]), zero(ũ))
            I = ifelse(open_face, neighborhood_range($(neighbor...), grid, c, projection.mask), (zero(ũ), zero(ũ)))
            return T, ĉ, I
        end

        @inline $face_state(i, j, k, grid::$FG, projection, U, c, s) = (zero(grid), zero(grid), (zero(grid), zero(grid)))
    end
end

# One pass of the active-set solution of Φ⁻¹ Σ w clamp(v - λ, I) = target, given the current shift λ
@inline function active_set_shift(λ, β, w, v, bound, targetΦ, decrease)
    active = map(βf -> ifelse(decrease, βf > λ, βf < λ), β)
    Wᵃ = sum(map((a, wf) -> ifelse(a, wf, zero(wf)), active, w))
    Sᵃ = sum(map((a, wf, vf, bf) -> wf * ifelse(a, vf, bf), active, w, v, bound))
    return ifelse(Wᵃ > 0, (Sᵃ - targetΦ) / max(Wᵃ, eps(typeof(λ))), λ)
end

@inline function ghost_cell_shift(i, j, k, grid, projection, U, c, Δt)
    FT = eltype(grid)
    cᵢ = @inbounds c[i, j, k]

    faces = (face_state_x(i, j, k, grid, projection, U, c, -1), face_state_x(i, j, k, grid, projection, U, c, +1),
             face_state_y(i, j, k, grid, projection, U, c, -1), face_state_y(i, j, k, grid, projection, U, c, +1),
             face_state_z(i, j, k, grid, projection, U, c, -1), face_state_z(i, j, k, grid, projection, U, c, +1))

    # Outflow weights, and outflow values clipped to their admissible intervals
    w = map(face -> max(face[1], zero(FT)), faces)
    v = map(face -> clamp(face[2], face[3]...), faces)
    Φ = sum(w) # throughput: outflow = inflow for a divergence-free transport
    ν = Δt * Φ / Vᶜᶜᶜ(i, j, k, grid)
    α = ifelse(ν > 0, max((1 - ν) / max(ν, eps(FT)), zero(FT)), zero(FT))

    cᵐⁱⁿ, cᵐᵃˣ = neighborhood_range(i, j, k, grid, c, projection.mask)
    cˡᵒ = cᵢ - α * (cᵐᵃˣ - cᵢ)
    cʰⁱ = cᵢ + α * (cᵢ - cᵐⁱⁿ)

    g₀ = sum(map(*, w, v)) / max(Φ, eps(FT))
    decrease = g₀ > cʰⁱ
    target = clamp(g₀, cˡᵒ, cʰⁱ)

    # g(λ) = Φ⁻¹ Σ w clamp(v - λ, I) is piecewise linear and non-increasing. Shifting towards `target`, face f
    # saturates at the end `bound` of its interval once λ passes β = v - bound. Solving the linear equation on the
    # faces that have not saturated, and dropping those that have, is exact after at most one pass per face.
    bound = map(face -> ifelse(decrease, face[3][1], face[3][2]), faces)
    β = map(-, v, bound)
    λ = zero(FT)
    for _ in 1:length(faces)
        λ = active_set_shift(λ, β, w, v, bound, target * Φ, decrease)
    end

    return ifelse((Φ > 0) & (Δt > 0) & (g₀ != target), λ, zero(FT))
end

@kernel function _compute_ghost_cell_shift!(projection, grid, U, c, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds projection.shift[i, j, k] = ghost_cell_shift(i, j, k, grid, projection, U, c, Δt)
end

"""
    update_ghost_cell_projection!(scheme, model, velocities, tracer, Δt)

Reconstruct the face values of `tracer` and compute the shifts of the band cells for the advecting `velocities`
and the time step `Δt` of the tracer flux divergence that follows.
"""
update_ghost_cell_projection!(scheme, model, velocities, tracer, Δt) = nothing

function update_ghost_cell_projection!(scheme::BoundedGhostCellWENO, model, velocities, tracer, Δt)
    grid = model.grid
    arch = architecture(grid)
    projection = scheme.buffer_scheme.projection

    compute_ghost_cell_face_values!(projection.faces, grid, scheme, velocities, tracer)

    isempty(projection.band) && return nothing
    Δt = convert(eltype(grid), Δt)
    launch!(arch, grid, :xyz, _compute_ghost_cell_shift!, projection, grid, velocities, tracer, Δt; active_cells_map = projection.band)
    fill_halo_regions!(projection.shift)

    return nothing
end
