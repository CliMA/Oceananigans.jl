using Oceananigans.Grids: Face, Periodic, RightFaceFolded, FaceExtendedTopology, SerialFoldedTopology
using Oceananigans.BoundaryConditions: West, East, South, North

#####
##### Which halo cells of a rank are copies of cells owned by another rank
#####
##### A rank owns its interior and the halos it fills itself: on bounded boundaries and in directions it holds alone.
##### Every side that communicates has one strip of halo: the west and east strips span the owned rows, the south and
##### north strips span the whole width, corners included. A strip receives the cells owned by other ranks through the
##### images of the grid: the periodic shifts and, on a tripolar grid, the mirror image through the fold.
#####

normal_dimension(::Union{West, East})   = 1
normal_dimension(::Union{South, North}) = 2

opposite_side(::West)  = East()
opposite_side(::East)  = West()
opposite_side(::South) = North()
opposite_side(::North) = South()

"""
    communicates(layout, rank, side, loc)

Return `true` if `rank` receives the halo on `side` of a field at `loc`ation from other ranks.
"""
function communicates(layout, rank, side, loc)
    d = normal_dimension(side)
    isnothing(loc[d]) && return false

    range = layout.ranges[rank + 1][d]
    N = layout.size[d]
    T = layout.topology[d]

    on_boundary = side isa Union{West, South} ? first(range) == 1 : last(range) == N
    periodic = T === Periodic && length(range) < N
    folded = side isa North && T <: SerialFoldedTopology && length(layout.ranges[rank + 1][1]) < layout.size[1]

    return !on_boundary || periodic || folded
end

# Faces have one more point than centers at a bounded (or face-folded) end
face_extension(layout, range, d, loc) = Int(loc[d] isa Face && last(range) == layout.size[d] && layout.topology[d] <: FaceExtendedTopology)

# Global indices of the parent array of `rank`; a reduced direction has a single index
function parent_range(layout, rank, d, H, loc)
    isnothing(loc[d]) && return 1:1
    range = layout.ranges[rank + 1][d]
    return first(range) - H : last(range) + face_extension(layout, range, d, loc) + H
end

function owned_range(layout, rank, d, H, loc)
    isnothing(loc[d]) && return 1:1
    range = layout.ranges[rank + 1][d]
    low_side, high_side = d == 1 ? (West(), East()) : (South(), North())
    first_owned = communicates(layout, rank, low_side, loc) ? first(range) : first(range) - H
    last_owned = last(range) + face_extension(layout, range, d, loc) + (communicates(layout, rank, high_side, loc) ? 0 : H)
    return first_owned:last_owned
end

# Offsets of the mirror image through the tripolar fold, (i, j) ↦ (oˣ - i, oʸ - j)
fold_offsets(layout, loc) = (layout.size[1] + 1 + layout.pivot_shift + (loc[1] isa Face),
                             2 * layout.size[2] + (layout.topology[2] === RightFaceFolded) + (loc[2] isa Face))

# On the fold, the north halo starts at the row that the fold maps onto itself, if there is one
function first_north_halo_row(layout, rank, loc)
    range = layout.ranges[rank + 1][2]
    if layout.topology[2] <: SerialFoldedTopology && last(range) == layout.size[2]
        _, oʸ = fold_offsets(layout, loc)
        return cld(oʸ, 2)
    else
        return last(range) + face_extension(layout, range, 2, loc) + 1
    end
end

function strip_rows(layout, rank, H, loc)
    rows = owned_range(layout, rank, 2, H[2], loc)
    communicates(layout, rank, North(), loc) || return rows
    return first(rows) : first_north_halo_row(layout, rank, loc) - 1
end

function halo_strip(layout, rank, ::West, H, loc)
    i = first(layout.ranges[rank + 1][1])
    return (i - H[1] : i - 1, strip_rows(layout, rank, H, loc))
end

function halo_strip(layout, rank, ::East, H, loc)
    i = last(layout.ranges[rank + 1][1])
    return (i + 1 : i + H[1], strip_rows(layout, rank, H, loc))
end

function halo_strip(layout, rank, ::South, H, loc)
    j = first(layout.ranges[rank + 1][2])
    return (parent_range(layout, rank, 1, H[1], loc), j - H[2] : j - 1)
end

function halo_strip(layout, rank, ::North, H, loc)
    j = first_north_halo_row(layout, rank, loc)
    return (parent_range(layout, rank, 1, H[1], loc), j : last(parent_range(layout, rank, 2, H[2], loc)))
end

"""
    HaloImage(region, σx, ox, σy, oy, sign)

The map `(i, j) ↦ (σx * i + ox, σy * j + oy)` from the halo cells in `region` to the cells they copy, multiplied by `sign`.
"""
struct HaloImage
    region :: NTuple{2, UnitRange{Int}}
    σx :: Int
    ox :: Int
    σy :: Int
    oy :: Int
    sign :: Int
end

preimage(r, σ, o) = σ == 1 ? (first(r) - o : last(r) - o) : (o - last(r) : o - first(r))
preimage(image::HaloImage, (i, j)) = (preimage(i, image.σx, image.ox), preimage(j, image.σy, image.oy))

function halo_images(layout, sender, loc, sign, H)
    Nx, Ny, _ = layout.size
    TX, TY, _ = layout.topology
    x, y, _ = layout.ranges[sender + 1]

    everywhere = (1 - H[1] : Nx + H[1] + 1, 1 - H[2] : Ny + H[2] + 1)
    x_shifts = TX === Periodic && length(x) < Nx ? (-Nx, 0, Nx) : (0,)
    y_shifts = TY === Periodic && length(y) < Ny ? (-Ny, 0, Ny) : (0,)

    TY <: SerialFoldedTopology || return [HaloImage(everywhere, 1, sx, 1, sy, 1) for sx in x_shifts for sy in y_shifts]

    # Cells beyond the fold copy their mirror image, and so do the cells of the fold row that come after their mirror
    oˣ, oʸ = fold_offsets(layout, loc)
    J = cld(oʸ, 2)
    below = (everywhere[1], first(everywhere[2]) : J - 1)
    above = (everywhere[1], J + iseven(oʸ) : last(everywhere[2]))

    fold_row = iseven(oʸ) ? (J:J) : (J:J-1)
    redundant = filter(i -> i > mod1(oˣ - i, Nx), 1:Nx)
    R = first(redundant):last(redundant)
    copied_row = [(R .+ p, fold_row) for p in (-Nx, 0, Nx)]
    kept_row = [(columns .+ p, fold_row) for columns in (1:first(R)-1, last(R)+1:Nx) for p in (-Nx, 0, Nx)]

    identity = [HaloImage(region, 1, sx, 1, 0, 1) for region in (below, kept_row...) for sx in x_shifts]
    mirror = [HaloImage(region, -1, oˣ + p, -1, oʸ, sign) for region in (above, copied_row...) for p in (-Nx, 0, Nx)]

    return vcat(identity, mirror)
end

# Halos span the whole vertical and the reduced directions, so ranks pair up only where they share that position
function same_position(layout, receiver, sender, loc)
    r = layout.ranges[receiver + 1]
    s = layout.ranges[sender + 1]
    return first(r[3]) ∈ s[3] && (!isnothing(loc[1]) || first(r[1]) ∈ s[1]) && (!isnothing(loc[2]) || first(r[2]) ∈ s[2])
end

"""
    halo_overlaps(layout, receiver, sender, loc, sign, H)

The halo cells of `receiver` that copy cells owned by `sender`, for a field at `loc`ation with halo `H`, as a list of
`(side, cells, image)` in global indices. Both ranks build the same list, so a message is tagged by its position.
"""
function halo_overlaps(layout, receiver, sender, loc, sign, H)
    overlaps = []
    same_position(layout, receiver, sender, loc) || return overlaps

    owned = (owned_range(layout, sender, 1, H[1], loc), owned_range(layout, sender, 2, H[2], loc))
    images = halo_images(layout, sender, loc, sign, H)

    for side in (West(), East(), South(), North())
        communicates(layout, receiver, side, loc) || continue
        strip = halo_strip(layout, receiver, side, H, loc)
        for image in images
            # On the fold row, the cells of a rank that do not copy their mirror are already in place
            receiver == sender && image.σx == 1 && continue
            cells = map(intersect, strip, image.region, preimage(image, owned))
            all(!isempty, cells) && push!(overlaps, (; side, cells, image))
        end
    end

    return overlaps
end
