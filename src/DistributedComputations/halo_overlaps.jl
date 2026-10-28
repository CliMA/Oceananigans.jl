using Oceananigans.Grids: Periodic, SerialFoldedTopology

#####
##### Which halo cells of a rank are copies of cells owned by another rank
#####
##### A rank owns its interior and its local halos: those on bounded boundaries and in directions it holds alone.
##### Every side that communicates has one strip of halo: the west and east strips span the owned rows, the south and
##### north strips span the whole width, corners included. A strip receives the cells owned by other ranks through the
##### images of the grid: the periodic copies and, on a tripolar grid, the mirror image through the fold.
#####

"""
    communicating_sides(connectivity, rank, loc)

The sides `(; west, east, south, north)` on which `rank` receives the halo of a field at `loc`ation from other ranks.
"""
function communicating_sides(connectivity, rank, loc)
    x, y, _ = connectivity.ranges[rank + 1]
    Nx, Ny, _ = connectivity.size
    TX, TY, _ = connectivity.topology

    periodic_x = TX === Periodic && length(x) < Nx
    periodic_y = TY === Periodic && length(y) < Ny
    folded = TY <: SerialFoldedTopology && length(x) < Nx

    west  = !isnothing(loc[1]) && (first(x) > 1  || periodic_x)
    east  = !isnothing(loc[1]) && (last(x)  < Nx || periodic_x)
    south = !isnothing(loc[2]) && (first(y) > 1  || periodic_y)
    north = !isnothing(loc[2]) && (last(y)  < Ny || periodic_y || folded)

    return (; west, east, south, north)
end

"""
    halo_geometry(connectivity, rank, halo, loc)

Global indices of the parent array of `rank`, of the cells it owns, and of the halo strip on each side that communicates
(`nothing` on the others), for a field at `loc`ation. A reduced direction has the single index 1.
"""
function halo_geometry(connectivity, rank, halo, loc)
    x, y, _ = connectivity.ranges[rank + 1]
    Nx, Ny, _ = connectivity.size
    TX, TY, _ = connectivity.topology
    Hx, Hy, _ = halo
    (; west, east, south, north) = communicating_sides(connectivity, rank, loc)

    last_x = last(x) == Nx ? length(loc[1], TX(), Nx) : last(x)
    last_y = last(y) == Ny ? length(loc[2], TY(), Ny) : last(y)

    parent_x = isnothing(loc[1]) ? (1:1) : (first(x) - Hx : last_x + Hx)
    parent_y = isnothing(loc[2]) ? (1:1) : (first(y) - Hy : last_y + Hy)

    owned_x = (west  ? first(x) : first(parent_x)) : (east  ? last_x : last(parent_x))
    owned_y = (south ? first(y) : first(parent_y)) : (north ? last_y : last(parent_y))

    # On the fold, the north strip also covers the last row, which the fold may map onto itself
    first_north_row = TY <: SerialFoldedTopology && last(y) == Ny ? last_y : last_y + 1
    strip_rows = north ? (first(owned_y) : first_north_row - 1) : owned_y

    strips = (west  = west  ? (first(parent_x) : first(x) - 1, strip_rows) : nothing,
              east  = east  ? (last_x + 1 : last(parent_x), strip_rows) : nothing,
              south = south ? (parent_x, first(parent_y) : first(y) - 1) : nothing,
              north = north ? (parent_x, first_north_row : last(parent_y)) : nothing)

    return (parent = (parent_x, parent_y), owned = (owned_x, owned_y), strips)
end

"""
    HaloImage(region, σ, ox, oy)

The halo cells `(i, j)` in `region` copy the cell `(σ * i + ox, σ * j + oy)`: `σ = 1` for a copy and `σ = -1` for the
mirror image through a fold.
"""
struct HaloImage
    region :: NTuple{2, UnitRange{Int}}
    σ :: Int
    ox :: Int
    oy :: Int
end

function halo_images(grid, loc)
    Nx, Ny, _ = architecture(grid).connectivity.size
    Hx, Hy, _ = halo_size(grid)
    return [HaloImage((1 - Hx : Nx + Hx + 1, 1 - Hy : Ny + Hy + 1), 1, 0, 0)]
end

"""
    halo_overlaps(grid, receiver, sender, loc)

The halo cells of `receiver` that copy cells owned by `sender`, for a field at `loc`ation on `grid`, as a list of
`(side, cells, σ, ox, oy)` in global indices: the `cells` on `side` copy `(σ * i + ox, σ * j + oy)`. Both ranks build
the same list, so a message is tagged by its position.
"""
function halo_overlaps(grid, receiver, sender, loc)
    connectivity = architecture(grid).connectivity
    Nx, Ny, _ = connectivity.size
    TX, TY, _ = connectivity.topology
    r = connectivity.ranges[receiver + 1]
    s = connectivity.ranges[sender + 1]
    overlaps = []

    # Halos span the whole vertical and the reduced directions, so ranks pair up only where they share those positions
    shares(d) = first(r[d]) ∈ s[d]
    paired = shares(3) && (!isnothing(loc[1]) || shares(1)) && (!isnothing(loc[2]) || shares(2))
    paired || return overlaps

    strips = halo_geometry(connectivity, receiver, halo_size(grid), loc).strips
    owned_x, owned_y = halo_geometry(connectivity, sender, halo_size(grid), loc).owned
    images = halo_images(grid, loc)

    # The periodic copies of the images, in the directions where the sender does not hold the whole domain
    x_shifts = TX === Periodic && length(s[1]) < Nx ? (-Nx, 0, Nx) : (0,)
    y_shifts = TY === Periodic && length(s[2]) < Ny ? (-Ny, 0, Ny) : (0,)

    for side in (:west, :east, :south, :north), image in images, Δx in x_shifts, Δy in y_shifts
        strip = strips[side]
        isnothing(strip) && continue

        σ = image.σ
        ox = image.ox + Δx
        oy = image.oy + Δy

        # A rank already holds its own copies, but not its own mirror images
        receiver == sender && σ == 1 && continue

        # The halo cells whose image is owned by the sender
        i = σ == 1 ? (first(owned_x) - ox : last(owned_x) - ox) : (ox - last(owned_x) : ox - first(owned_x))
        j = σ == 1 ? (first(owned_y) - oy : last(owned_y) - oy) : (oy - last(owned_y) : oy - first(owned_y))
        cells = (intersect(strip[1], image.region[1], i), intersect(strip[2], image.region[2], j))

        all(!isempty, cells) && push!(overlaps, (; side, cells, σ, ox, oy))
    end

    return overlaps
end
