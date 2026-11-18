#####
##### Tables that interpolate receiver nodes from the interior nodes of a Yin-Yang panel: the halo nodes of
##### the other panel, or the nodes of a grid in geographic coordinates. They depend only on the index space,
##### which is the same on both panels, and are built once on the CPU in Float64.
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

Base.length(table::YinYangInterpolationTable) = length(table.receivers.i)

map_table(f, table::YinYangInterpolationTable{S}) where S =
    YinYangInterpolationTable{S}(f(table.receivers), f(table.donors), f(table.weights), f(table.rotation))

Adapt.adapt_structure(to, table::YinYangInterpolationTable) = map_table(x -> adapt(to, x), table)
Grids.on_architecture(arch, table::YinYangInterpolationTable) = map_table(x -> on_architecture(arch, x), table)

# The Lagrange weight at the fractional index x★ of node `a` of the s nodes i₀, i₀ + 1, …, i₀ + s - 1.
lagrange_weight(x★, i₀, s, a) = prod((x★ - (i₀ + b - 1)) / (a - b) for b in 1:s if b != a; init = one(x★))

# The table of the `receivers` interpolated from the `s × s` stencils of the `Nx × Ny` interior nodes of a
# panel that bracket the receivers' `fractional_indices`, with the rotation angles `ψ` from the panel's basis.
function YinYangInterpolationTable(arch, FT, s, receivers, fractional_indices, ψ, (Nx, Ny))
    starts = [interior_stencil_start(i★, j★, s, (Nx, Ny)) for (i★, j★) in fractional_indices]
    Nmissing = count(isnothing, starts)
    Nmissing == 0 || throw(ArgumentError("$Nmissing receivers have no $s×$s stencil of interior donors: " *
                                         "the YinYangGrid overlap is too small."))

    N = length(receivers)
    weights(d) = FT[lagrange_weight(fractional_indices[n][d], starts[n][d], s, a) for n in 1:N, a in 1:s]
    table = YinYangInterpolationTable{s}((; i = first.(receivers), j = last.(receivers)),
                                         (; i = first.(starts), j = last.(starts)),
                                         (; x = weights(1), y = weights(2)),
                                         (; cosψ = FT.(cos.(ψ)), sinψ = FT.(sin.(ψ))))
    return on_architecture(arch, table)
end

# Tables for the halo nodes of a panel: scalars at the four horizontal locations, and each horizontal velocity
# component from the location of the other. All tables list the same receivers in the same order.
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

# The table that interpolates donors at `(ℓx′, ℓy′)` to the receivers at `(ℓx, ℓy)`.
@inline interpolation_table(t::YinYangInterpolation, ::Center, ::Center, ::Center, ::Center) = t.center_center
@inline interpolation_table(t::YinYangInterpolation, ::Face,   ::Center, ::Face,   ::Center) = t.face_center
@inline interpolation_table(t::YinYangInterpolation, ::Center, ::Face,   ::Center, ::Face)   = t.center_face
@inline interpolation_table(t::YinYangInterpolation, ::Face,   ::Face,   ::Face,   ::Face)   = t.face_face
@inline interpolation_table(t::YinYangInterpolation, ::Face,   ::Center, ::Center, ::Face)   = t.face_center_from_center_face
@inline interpolation_table(t::YinYangInterpolation, ::Center, ::Face,   ::Face,   ::Center) = t.center_face_from_face_center

@inline interpolation_table(t::YinYangInterpolation, ℓx, ℓy) = interpolation_table(t, ℓx, ℓy, ℓx, ℓy)

# The tables that interpolate the halo nodes of a panel with cell faces at `(λᶠ, φᶠ)` and horizontal `halo`
# from the interior nodes of the other panel with `s × s` stencils
function YinYangInterpolation(arch, FT, (λᶠ, φᶠ), halo, s)
    frame = panel_frame(λᶠ, φᶠ)
    interior_size = (length(λᶠ) - 1, length(φᶠ) - 1)
    receivers = halo_nodes(interior_size, halo)

    function table(receiver_location, donor_location)
        donors = [donor_fractional_indices(i, j, frame, receiver_location, donor_location) for (i, j) in receivers]
        fractional_indices = [(i★, j★) for (i★, j★, _, _) in donors]
        ψ = [yin_yang_rotation_angle(λ″, φ″) for (_, _, λ″, φ″) in donors]
        return YinYangInterpolationTable(arch, FT, s, receivers, fractional_indices, ψ, interior_size)
    end

    F, C = Face(), Center()
    return YinYangInterpolation(table((C, C), (C, C)), table((F, C), (F, C)), table((C, F), (C, F)),
                                table((F, F), (F, F)), table((F, C), (C, F)), table((C, F), (F, C)))
end
