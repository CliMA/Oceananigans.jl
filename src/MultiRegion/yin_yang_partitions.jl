#####
##### The halos of the subregions of the Yin-Yang panels (see `SubregionPartition`). Every halo node of a subregion is
##### either a copy of an interior node of a sibling, or interpolated from the interior of the other panel. The
##### interpolation stencils of the panel tables are split by the donor subregion that holds them, and read the
##### halo-filled arrays of that subregion: copies are filled first.
#####

# Region 1 is the Yin panel and region 2 is the Yang panel
struct YinYangPartition <: AbstractPartition end

Base.length(::YinYangPartition) = 2
Base.summary(::YinYangPartition) = "YinYangPartition with (1: Yin, 2: Yang)"

# The halo of subregion r at one location: `copies` and `interpolations` are Vectors of `(donor, table)`
struct YinYangRegionHalo{C, I}
    copies :: C
    interpolations :: I
end

# The halo of the horizontal vector of subregion r: `u` and `v` are Vectors of `(donor, (table, cross_table))`
struct YinYangRegionVectorHalo{C, U}
    u_copies :: C
    v_copies :: C
    u :: U
    v :: U
end

struct YinYangRegionHalos{S, V}
    scalars :: S # (; center_center, face_center, center_face, face_face) of `YinYangRegionHalo`
    vector :: V  # `YinYangRegionVectorHalo`
end

on_architecture_pieces(arch, pieces) = [(donor, on_architecture(arch, tables)) for (donor, tables) in pieces]

Grids.on_architecture(arch, h::YinYangRegionHalo) =
    YinYangRegionHalo(on_architecture_pieces(arch, h.copies), on_architecture_pieces(arch, h.interpolations))

Grids.on_architecture(arch, h::YinYangRegionVectorHalo) =
    YinYangRegionVectorHalo((on_architecture_pieces(arch, getfield(h, name)) for name in fieldnames(YinYangRegionVectorHalo))...)

Grids.on_architecture(arch, h::YinYangRegionHalos) =
    YinYangRegionHalos(map(s -> on_architecture(arch, s), h.scalars), on_architecture(arch, h.vector))

# Rows `rows` of `table`, with receivers and donors shifted by the offsets of their subregions
function table_rows(table::YinYangInterpolationTable{S}, rows, receiver_offsets, donor_offsets) where S
    receivers = (; i = table.receivers.i[rows] .- receiver_offsets[1], j = table.receivers.j[rows] .- receiver_offsets[2])
    donors = (; i = table.donors.i[rows] .- donor_offsets[1], j = table.donors.j[rows] .- donor_offsets[2])
    weights = (; x = table.weights.x[rows, :], y = table.weights.y[rows, :])
    rotation = (; cosψ = table.rotation.cosψ[rows], sinψ = table.rotation.sinψ[rows])
    return YinYangInterpolationTable{S}(receivers, donors, weights, rotation)
end

# The subregion of `panel` that holds, halos included, the stencils of row n of every one of `tables`
function donor_subregion(layout, panel, tables, n, (Hx, Hy))
    s = size(first(tables).weights.x, 2)
    I₀ = maximum(t.donors.i[n] for t in tables)
    J₀ = maximum(t.donors.j[n] for t in tables)
    d = owning_subregion(layout, panel, (I₀, J₀))
    (dox, doy), (dnx, dny) = layout[d].offsets, layout[d].size
    all(stencil_held(t.donors.i[n] - dox, s, dnx, Hx) && stencil_held(t.donors.j[n] - doy, s, dny, Hy) for t in tables) ||
        throw(ArgumentError("The subregions of the YinYangGrid are too small for its halo and stencil width."))
    return d
end

# The panel `tables` list the panel halo nodes as receivers, in the same order
function region_halo_pieces(layout, r, tables, panel_size, halo)
    copies, seam_nodes = subregion_halos(layout, r, panel_size, halo)
    table = first(tables)
    row = Dict((table.receivers.i[n], table.receivers.j[n]) => n for n in 1:length(table))
    interpolated = Dict{Int, Vector{Int}}()

    for node in seam_nodes
        n = row[node]
        d = donor_subregion(layout, 3 - layout[r].region, tables, n, halo)
        push!(get!(interpolated, d, Int[]), n)
    end

    # A Vector, not a Tuple: the number of donors varies by subregion and must not enter the type of the grid
    interpolations = [(d, map(t -> table_rows(t, rows, layout[r].offsets, layout[d].offsets), tables))
                      for (d, rows) in sort(collect(interpolated))]
    return copies, interpolations
end

function YinYangRegionHalos(layout, r, interpolation::YinYangInterpolation, panel_size, halo)
    scalar(t) = YinYangRegionHalo(region_halo_pieces(layout, r, (t,), panel_size, halo)...)
    scalars = (; center_center = scalar(interpolation.center_center), face_center = scalar(interpolation.face_center),
                 center_face = scalar(interpolation.center_face), face_face = scalar(interpolation.face_face))

    u_copies, u = region_halo_pieces(layout, r, (interpolation.face_center, interpolation.face_center_from_center_face),
                                     panel_size, halo)
    v_copies, v = region_halo_pieces(layout, r, (interpolation.center_face_from_face_center, interpolation.center_face),
                                     panel_size, halo)

    return YinYangRegionHalos(scalars, YinYangRegionVectorHalo(u_copies, v_copies, u, v))
end

@inline region_halo(h::YinYangRegionHalos, ::Center, ::Center) = h.scalars.center_center
@inline region_halo(h::YinYangRegionHalos, ::Face,   ::Center) = h.scalars.face_center
@inline region_halo(h::YinYangRegionHalos, ::Center, ::Face)   = h.scalars.center_face
@inline region_halo(h::YinYangRegionHalos, ::Face,   ::Face)   = h.scalars.face_face

# The owned area fractions of subregion r, cut from those of its panel
function region_cell_weights(layout, r, panel_weights, (Hx, Hy))
    (ox, oy), (nx, ny) = layout[r].offsets, layout[r].size
    w = StaticStridesOffsetArray(zeros(eltype(panel_weights), nx + 2Hx, ny + 2Hy, 1), -Hx, -Hy, 0)
    for j in 1-Hy:ny+Hy, i in 1-Hx:nx+Hx
        w[i, j, 1] = panel_weights[i + ox, j + oy, 1]
    end
    return w
end
