#####
##### Tables that interpolate the halo nodes of a Yin-Yang panel from the interior nodes of the other panel.
##### They depend only on the index space, which is the same on both panels, so one table serves Yin → Yang
##### and Yang → Yin alike. They are built once on the CPU in Float64.
#####

# Receiver `n` is interpolated as Σₐ Σ_b weights.x[n, a] weights.y[n, b] donor[i₀ + a - 1, j₀ + b - 1, k]
# from the S × S stencil that starts at (i₀, j₀) = (donors.i[n], donors.j[n]).
struct YinYangInterpolationTable{S, R, W, A}
    receivers :: R # (; i, j): the receiver nodes
    donors :: R    # (; i, j): the first node of each receiver's S × S donor stencil
    weights :: W   # (; x, y): the Lagrange weights of each receiver, of size (receivers, S)
    rotation :: A  # (; cosψ, sinψ): the rotation from the donor panel's basis to the receiver's at each receiver

    YinYangInterpolationTable{S}(receivers::R, donors::R, weights::W, rotation::A) where {S, R, W, A} =
        new{S, R, W, A}(receivers, donors, weights, rotation)
end

stencil_width(::YinYangInterpolationTable{S}) where S = S
Base.length(table::YinYangInterpolationTable) = length(table.receivers.i)

map_table(f, table::YinYangInterpolationTable{S}) where S =
    YinYangInterpolationTable{S}(f(table.receivers), f(table.donors), f(table.weights), f(table.rotation))

Adapt.adapt_structure(to, table::YinYangInterpolationTable) = map_table(x -> adapt(to, x), table)
Grids.on_architecture(arch, table::YinYangInterpolationTable) = map_table(x -> on_architecture(arch, x), table)

Base.summary(table::YinYangInterpolationTable{S}) where S =
    string("YinYangInterpolationTable with ", length(table), " receivers and ", S, "×", S, " stencils")

Base.show(io::IO, table::YinYangInterpolationTable) = print(io, summary(table))

# The Lagrange weights at the fractional index x★ of the s nodes i₀, i₀ + 1, …, i₀ + s - 1.
lagrange_weights(x★, i₀, s) = [prod((x★ - (i₀ + b - 1)) / (a - b) for b in 1:s if b != a; init = one(x★)) for a in 1:s]

# The table of the `receivers` interpolated from the `s × s` donor stencils that start at `starts` around
# the fractional indices `fractional_indices`, with the basis rotation angles `ψ` from the donor panel.
function YinYangInterpolationTable(arch, FT, s, receivers, starts, fractional_indices, ψ)
    N = length(receivers)
    x_weights = zeros(FT, N, s)
    y_weights = zeros(FT, N, s)
    for n in 1:N
        x_weights[n, :] .= lagrange_weights(fractional_indices[n][1], starts[n][1], s)
        y_weights[n, :] .= lagrange_weights(fractional_indices[n][2], starts[n][2], s)
    end

    table = YinYangInterpolationTable{s}((; i = first.(receivers), j = last.(receivers)),
                                         (; i = first.(starts), j = last.(starts)),
                                         (; x = x_weights, y = y_weights),
                                         (; cosψ = FT.(cos.(ψ)), sinψ = FT.(sin.(ψ))))
    return on_architecture(arch, table)
end

# The table of the halo nodes at `receiver_location` of a panel with cell faces at `(λᶠ, φᶠ)`, interpolated
# from the interior nodes of the other panel at `donor_location`.
function yin_yang_interpolation_table(arch, FT, (λᶠ, φᶠ), halo, s, receiver_location, donor_location)
    frame = panel_frame(λᶠ, φᶠ)
    interior_size = (length(λᶠ) - 1, length(φᶠ) - 1)
    receivers = halo_nodes(interior_size, halo)
    donors = [donor_fractional_indices(i, j, frame, receiver_location, donor_location) for (i, j) in receivers]
    fractional_indices = [(i★, j★) for (i★, j★, _, _) in donors]
    starts = [interior_stencil_start(i★, j★, s, interior_size) for (i★, j★) in fractional_indices]

    Nmissing = count(isnothing, starts)
    Nmissing == 0 || throw(ArgumentError("The YinYangGrid overlap is too small for $s×$s stencils: " *
                                         "$Nmissing halo nodes have no interior donors."))
    ψ = [yin_yang_rotation_angle(λ″, φ″) for (_, _, λ″, φ″) in donors]
    return YinYangInterpolationTable(arch, FT, s, receivers, starts, fractional_indices, ψ)
end

# Tables for scalars at the four horizontal locations, and for each horizontal velocity component at the
# location of the other. All tables list the same receivers in the same order.
struct YinYangInterpolation{T}
    center_center :: T                # (Center, Center) receivers from (Center, Center) donors
    face_center :: T                  # (Face, Center) receivers from (Face, Center) donors
    center_face :: T                  # (Center, Face) receivers from (Center, Face) donors
    face_face :: T                    # (Face, Face) receivers from (Face, Face) donors
    face_center_from_center_face :: T # (Face, Center) receivers from (Center, Face) donors
    center_face_from_face_center :: T # (Center, Face) receivers from (Face, Center) donors
end

Grids.on_architecture(arch, interpolation::YinYangInterpolation) =
    YinYangInterpolation((on_architecture(arch, getfield(interpolation, name)) for name in fieldnames(YinYangInterpolation))...)

stencil_width(interpolation::YinYangInterpolation) = stencil_width(interpolation.center_center)

Base.summary(interpolation::YinYangInterpolation) =
    string("YinYangInterpolation with ", length(interpolation.center_center), " receivers per location and ",
           stencil_width(interpolation), "×", stencil_width(interpolation), " stencils")

Base.show(io::IO, interpolation::YinYangInterpolation) = print(io, summary(interpolation))

# The table that interpolates donors at `(ℓx′, ℓy′)` to the receivers at `(ℓx, ℓy)`.
@inline interpolation_table(t::YinYangInterpolation, ::Center, ::Center, ::Center, ::Center) = t.center_center
@inline interpolation_table(t::YinYangInterpolation, ::Face,   ::Center, ::Face,   ::Center) = t.face_center
@inline interpolation_table(t::YinYangInterpolation, ::Center, ::Face,   ::Center, ::Face)   = t.center_face
@inline interpolation_table(t::YinYangInterpolation, ::Face,   ::Face,   ::Face,   ::Face)   = t.face_face
@inline interpolation_table(t::YinYangInterpolation, ::Face,   ::Center, ::Center, ::Face)   = t.face_center_from_center_face
@inline interpolation_table(t::YinYangInterpolation, ::Center, ::Face,   ::Face,   ::Center) = t.center_face_from_face_center

@inline interpolation_table(t::YinYangInterpolation, ℓx, ℓy) = interpolation_table(t, ℓx, ℓy, ℓx, ℓy)

# The interpolation of the halos of both panels with cell faces at `panel_faces`, horizontal `halo` and
# `stencil_width × stencil_width` stencils
function YinYangInterpolation(arch, FT, panel_faces, halo, stencil_width)
    table(receiver_location, donor_location) =
        yin_yang_interpolation_table(arch, FT, panel_faces, halo, stencil_width, receiver_location, donor_location)

    F, C = Face(), Center()
    return YinYangInterpolation(table((C, C), (C, C)), table((F, C), (F, C)), table((C, F), (C, F)),
                                table((F, F), (F, F)), table((F, C), (C, F)), table((C, F), (F, C)))
end
