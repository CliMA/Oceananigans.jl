####
#### Envelope formulations for the multi-envelope vertical coordinate
####

# A formulation is the *generator* of the static envelope metric: it supplies the resting envelope
# depths and hence σᵉ = ∂ẑ/∂r and the physical resting column depth h. Skeleton instances (envelope
# fields = `nothing`) are built before the grid exists; `materialize_envelopes!` fills them once the
# horizontal nodes are known (the skeleton → materialize pattern, after Breeze).

abstract type AbstractEnvelopeFormulation end

"""
    LinearEnvelope(; bottom=nothing)

Single bottom envelope with a linear (pure-σ) stretch: the resting depth maps as ẑ(r) = r · H(x,y)/Lz,
so σᵉ = ∂ẑ/∂r = H/Lz is depth-independent and h = H. `bottom` holds the resting envelope depth H(x,y)
(filled by [`materialize_envelopes!`](@ref)).
"""
struct LinearEnvelope{B} <: AbstractEnvelopeFormulation
    bottom :: B
end

LinearEnvelope(; bottom=nothing) = LinearEnvelope(bottom)

# Allocate the formulation's envelope fields before the grid exists (mirrors `new_data` of the σ fields).
allocate_envelope_formulation(::Nothing, FT, arch, sz, halo, topo) = nothing

function allocate_envelope_formulation(f::LinearEnvelope, FT, arch, sz, halo, topo)
    f.bottom !== nothing && return Architectures.on_architecture(arch, f)
    bottom = new_data(FT, arch, (Center, Center, Nothing), topo, sz, halo)
    fill!(bottom, 0)
    return LinearEnvelope(bottom)
end

Adapt.adapt_structure(to, f::LinearEnvelope) = LinearEnvelope(Adapt.adapt(to, f.bottom))
Architectures.on_architecture(arch, f::LinearEnvelope) = LinearEnvelope(on_architecture(arch, f.bottom))

"""
    MultiEnvelope(; level_counts)

Multi-envelope formulation: `n = length(level_counts)` envelopes split the column into `n` sub-zones
(surface zone first), with `level_counts[i]` reference levels in zone `i` (Σ = Nz). Within each zone the
resting map is linear between the bounding envelopes, so σᵉ = (physical zone thickness)/(reference zone
thickness) is constant in the zone and varies across zones (depth-dependent). The envelope depths are
filled by [`materialize_envelopes!`](@ref) with a tuple of `n` depth functions (shallowest → deepest).

Vertical refinement *inside* a zone comes from the reference `r_faces`; the cubic-spline C¹ transition
zones of Bruciaferri et al. (2018, Eq. 12) are a future refinement (interfaces are C⁰ here).
"""
struct MultiEnvelope{E, N} <: AbstractEnvelopeFormulation
    envelopes    :: E   # NTuple of n (Center, Center, Nothing) envelope-depth fields, shallowest → deepest
    level_counts :: N   # NTuple of n level counts, surface zone first
end

MultiEnvelope(; level_counts) = MultiEnvelope(nothing, level_counts)

function allocate_envelope_formulation(f::MultiEnvelope, FT, arch, sz, halo, topo)
    f.envelopes !== nothing && return Architectures.on_architecture(arch, f)
    n = length(f.level_counts)
    envelopes = ntuple(n) do _
        envelope = new_data(FT, arch, (Center, Center, Nothing), topo, sz, halo)
        fill!(envelope, 0)
        envelope
    end
    return MultiEnvelope(envelopes, f.level_counts)
end

Adapt.adapt_structure(to, f::MultiEnvelope) =
    MultiEnvelope(map(e -> Adapt.adapt(to, e), f.envelopes), f.level_counts)

Architectures.on_architecture(arch, f::MultiEnvelope) =
    MultiEnvelope(map(e -> on_architecture(arch, e), f.envelopes), f.level_counts)

"""
    materialize_envelopes!(grid, bottom_height)

Fill the static envelope metric (σᵉ and the resting column depth h, at every stagger) of a
`MultiEnvelopeVerticalDiscretization` grid in place from `bottom_height(x, y)` (the resting depth of the
deepest envelope, positive). Must be called after the grid is built and before constructing a model.

The method is defined in the `Fields` module, where `Field`/`set!`/`fill_halo_regions!` are available.
"""
function materialize_envelopes! end
