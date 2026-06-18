####
#### Multi-envelope vertical coordinate (ME*s*, Bruciaferri et al. 2018)
####

# The physical level depth factorises as
#
#     z(i, j, k, t) = r(k) · σᵉ(i, j, k) · σ(i, j, t) + η(i, j, t)
#
# with σᵉ the *static*, depth-dependent envelope Jacobian (∂ẑ/∂r of the resting multi-envelope map) and
# σ the *column-uniform*, time-varying z-star scaling (H + η)/H. The z-star fields are identical to those
# of `MutableVerticalDiscretization` and evolved by the unmodified z-star machinery; σᵉ is precomputed once.

"""
    struct MultiEnvelopeVerticalDiscretization{...} <: AbstractMutableVerticalDiscretization

Multi-envelope terrain/pycnocline-following vertical coordinate that rides the free surface. Combines a
static, depth-dependent envelope Jacobian `σ_env` with the column-uniform z-star scaling `σ_fs`.

Fields
======

$(FIELDS)
"""
struct MultiEnvelopeVerticalDiscretization{C, D, E, F, H, CC, FC, CF, FF, SE, HR, FM} <: AbstractMutableVerticalDiscretization
    "Face-centered reference coordinate"
    cᵃᵃᶠ :: C
    "Cell-centered reference coordinate"
    cᵃᵃᶜ :: D
    "Face-centered reference grid spacing"
    Δᵃᵃᶠ :: E
    "Cell-centered reference grid spacing"
    Δᵃᵃᶜ :: F
    "Surface elevation at the current time step"
    ηⁿ :: H
    "(Center, Center) z-star scaling at the current time step"
    σᶜᶜⁿ :: CC
    "(Face, Center) z-star scaling at the current time step"
    σᶠᶜⁿ :: FC
    "(Center, Face) z-star scaling at the current time step"
    σᶜᶠⁿ :: CF
    "(Face, Face) z-star scaling at the current time step"
    σᶠᶠⁿ :: FF
    "(Center, Center) z-star scaling at the previous time step"
    σᶜᶜ⁻ :: CC
    "Time derivative of the cell-centered z-star scaling"
    ∂t_σ :: CC
    "(Center, Center) static envelope Jacobian ∂ẑ/∂r"
    σᶜᶜᵉ :: SE
    "(Face, Center) static envelope Jacobian"
    σᶠᶜᵉ :: SE
    "(Center, Face) static envelope Jacobian"
    σᶜᶠᵉ :: SE
    "(Face, Face) static envelope Jacobian"
    σᶠᶠᵉ :: SE
    "(Center, Center, Center) precomputed resting (σ_fs=1) physical znode −Σ_{k′>k}Δr σᵉ − ½Δr σᵉ (for O(1) masking)"
    zᶜᶜᶜᵉ :: SE
    "(Center, Center) physical resting column depth Σ Δr σᵉ"
    hᶜᶜ :: HR
    "(Face, Center) physical resting column depth"
    hᶠᶜ :: HR
    "(Center, Face) physical resting column depth"
    hᶜᶠ :: HR
    "(Face, Face) physical resting column depth"
    hᶠᶠ :: HR
    "Envelope generator (formulation), or `nothing` for a plain stretched grid"
    formulation :: FM
end

"""
    MultiEnvelopeVerticalDiscretization(r_faces; formulation=nothing)

Construct a `MultiEnvelopeVerticalDiscretization` from reference (computational) `r_faces`, which may be a
`Tuple`, a function of an index `k`, or an `AbstractArray`. The field arrays are allocated later by
`generate_coordinate` once the horizontal grid size is known; the static envelope metric is then filled by
the `formulation` (or left at `σ_env = 1`, a plain stretched z-star grid, when `formulation === nothing`).
"""
MultiEnvelopeVerticalDiscretization(r_faces; formulation=nothing) =
    MultiEnvelopeVerticalDiscretization(r_faces, r_faces, (nothing for i in 1:18)..., formulation)

const RegularMultiEnvelopeVerticalDiscretization = MultiEnvelopeVerticalDiscretization{<:Any, <:Any, <:Number}

coordinate_summary(::Bounded, z::RegularMultiEnvelopeVerticalDiscretization, name) =
    @sprintf("regularly spaced with multi-envelope Δr=%s", prettysummary(z.Δᵃᵃᶜ))

coordinate_summary(::Bounded, z::MultiEnvelopeVerticalDiscretization, name) =
    @sprintf("variably spaced with multi-envelope min(Δr)=%s, max(Δr)=%s",
             prettysummary(minimum(parent(z.Δᵃᵃᶜ))),
             prettysummary(maximum(parent(z.Δᵃᵃᶜ))))

function Base.show(io::IO, z::MultiEnvelopeVerticalDiscretization)
    print(io, "MultiEnvelopeVerticalDiscretization with reference interfaces r:\n")
    Base.show(io, z.cᵃᵃᶠ)
end

#####
##### Coordinate generation
#####

generate_coordinate(FT, ::Periodic, N, H, ::MultiEnvelopeVerticalDiscretization, coordinate_name, arch, args...) =
    throw(ArgumentError("Periodic domains are not supported for MultiEnvelopeVerticalDiscretization"))

function generate_coordinate(FT, topo, size, halo, coordinate::MultiEnvelopeVerticalDiscretization, coordinate_name, dim::Int, arch)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    if dim != 3
        throw(ArgumentError("MultiEnvelopeVerticalDiscretization is supported only in the third dimension (z)"))
    end

    if coordinate_name != :z
        throw(ArgumentError("MultiEnvelopeVerticalDiscretization is supported only for the z-coordinate"))
    end

    r_faces = coordinate.cᵃᵃᶠ

    LR, rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ = generate_coordinate(FT, topo[3](), Nz, Hz, r_faces, :r, arch)

    args = (topo, (Nx, Ny, Nz), (Hx, Hy, Hz))

    σᶜᶜ⁻ = new_data(FT, arch, (Center, Center, Nothing), args...)
    σᶜᶜⁿ = new_data(FT, arch, (Center, Center, Nothing), args...)
    σᶠᶜⁿ = new_data(FT, arch, (Face,   Center, Nothing), args...)
    σᶜᶠⁿ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    σᶠᶠⁿ = new_data(FT, arch, (Face,   Face,   Nothing), args...)
    ηⁿ   = new_data(FT, arch, (Center, Center, Nothing), args...)
    ∂t_σ = new_data(FT, arch, (Center, Center, Nothing), args...)

    σᶜᶜᵉ = new_data(FT, arch, (Center, Center, Center), args...)
    σᶠᶜᵉ = new_data(FT, arch, (Face,   Center, Center), args...)
    σᶜᶠᵉ = new_data(FT, arch, (Center, Face,   Center), args...)
    σᶠᶠᵉ = new_data(FT, arch, (Face,   Face,   Center), args...)
    zᶜᶜᶜᵉ = new_data(FT, arch, (Center, Center, Center), args...)   # precomputed resting znode (filled by materialize_envelopes!)

    hᶜᶜ = new_data(FT, arch, (Center, Center, Nothing), args...)
    hᶠᶜ = new_data(FT, arch, (Face,   Center, Nothing), args...)
    hᶜᶠ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    hᶠᶠ = new_data(FT, arch, (Face,   Face,   Nothing), args...)

    for σ in (σᶜᶜ⁻, σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜᵉ, σᶠᶜᵉ, σᶜᶠᵉ, σᶠᶠᵉ)
        fill!(σ, 1)
    end
    fill!(zᶜᶜᶜᵉ, 0)

    # With σᵉ = 1 the physical resting depth equals the reference column extent; a `formulation`
    # overwrites both σᵉ and h via `compute_envelope_metric!` once the grid exists.
    for h in (hᶜᶜ, hᶠᶜ, hᶜᶠ, hᶠᶠ)
        fill!(h, LR)
    end

    formulation = allocate_envelope_formulation(coordinate.formulation, FT, arch, (Nx, Ny, Nz), (Hx, Hy, Hz), topo)

    coordinate = MultiEnvelopeVerticalDiscretization(rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ, ηⁿ,
                                                     σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, ∂t_σ,
                                                     σᶜᶜᵉ, σᶠᶜᵉ, σᶜᶠᵉ, σᶠᶠᵉ, zᶜᶜᶜᵉ,
                                                     hᶜᶜ, hᶠᶜ, hᶜᶠ, hᶠᶠ,
                                                     formulation)

    return LR, coordinate
end

#####
##### Adapt and on_architecture
#####

Adapt.adapt_structure(to, coord::MultiEnvelopeVerticalDiscretization) =
    MultiEnvelopeVerticalDiscretization(Adapt.adapt(to, coord.cᵃᵃᶠ),
                                        Adapt.adapt(to, coord.cᵃᵃᶜ),
                                        Adapt.adapt(to, coord.Δᵃᵃᶠ),
                                        Adapt.adapt(to, coord.Δᵃᵃᶜ),
                                        Adapt.adapt(to, coord.ηⁿ),
                                        Adapt.adapt(to, coord.σᶜᶜⁿ),
                                        Adapt.adapt(to, coord.σᶠᶜⁿ),
                                        Adapt.adapt(to, coord.σᶜᶠⁿ),
                                        Adapt.adapt(to, coord.σᶠᶠⁿ),
                                        Adapt.adapt(to, coord.σᶜᶜ⁻),
                                        Adapt.adapt(to, coord.∂t_σ),
                                        Adapt.adapt(to, coord.σᶜᶜᵉ),
                                        Adapt.adapt(to, coord.σᶠᶜᵉ),
                                        Adapt.adapt(to, coord.σᶜᶠᵉ),
                                        Adapt.adapt(to, coord.σᶠᶠᵉ),
                                        Adapt.adapt(to, coord.zᶜᶜᶜᵉ),
                                        Adapt.adapt(to, coord.hᶜᶜ),
                                        Adapt.adapt(to, coord.hᶠᶜ),
                                        Adapt.adapt(to, coord.hᶜᶠ),
                                        Adapt.adapt(to, coord.hᶠᶠ),
                                        Adapt.adapt(to, coord.formulation))

Architectures.on_architecture(arch, coord::MultiEnvelopeVerticalDiscretization) =
    MultiEnvelopeVerticalDiscretization(on_architecture(arch, coord.cᵃᵃᶠ),
                                        on_architecture(arch, coord.cᵃᵃᶜ),
                                        on_architecture(arch, coord.Δᵃᵃᶠ),
                                        on_architecture(arch, coord.Δᵃᵃᶜ),
                                        on_architecture(arch, coord.ηⁿ),
                                        on_architecture(arch, coord.σᶜᶜⁿ),
                                        on_architecture(arch, coord.σᶠᶜⁿ),
                                        on_architecture(arch, coord.σᶜᶠⁿ),
                                        on_architecture(arch, coord.σᶠᶠⁿ),
                                        on_architecture(arch, coord.σᶜᶜ⁻),
                                        on_architecture(arch, coord.∂t_σ),
                                        on_architecture(arch, coord.σᶜᶜᵉ),
                                        on_architecture(arch, coord.σᶠᶜᵉ),
                                        on_architecture(arch, coord.σᶜᶠᵉ),
                                        on_architecture(arch, coord.σᶠᶠᵉ),
                                        on_architecture(arch, coord.zᶜᶜᶜᵉ),
                                        on_architecture(arch, coord.hᶜᶜ),
                                        on_architecture(arch, coord.hᶠᶜ),
                                        on_architecture(arch, coord.hᶜᶠ),
                                        on_architecture(arch, coord.hᶠᶠ),
                                        on_architecture(arch, coord.formulation))

#####
##### Utilities
#####

function validate_dimension_specification(T, ξ::MultiEnvelopeVerticalDiscretization, dir, N, FT)
    cᶠ = validate_dimension_specification(T, ξ.cᵃᵃᶠ, dir, N, FT)
    cᶜ = validate_dimension_specification(T, ξ.cᵃᵃᶜ, dir, N, FT)
    args = Tuple(getproperty(ξ, prop) for prop in propertynames(ξ))
    return MultiEnvelopeVerticalDiscretization(cᶠ, cᶜ, args[3:end]...)
end

#####
##### Grid alias and physical resting column depth
#####
#####
##### `static_column_depthᶜᶜᵃ` must return the *physical* resting depth Σ Δr σ_env (not the reference
##### extent `Lz`) so the z-star closure ∂t_σ_fs = -∇·U / H uses the correct column depth. Immersed grids
##### override these again in `grid_fitted_bottom.jl`.

const MultiEnvelopeGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded, <:MultiEnvelopeVerticalDiscretization}

@inline static_column_depthᶜᶜᵃ(i, j, grid::MultiEnvelopeGrid) = @inbounds grid.z.hᶜᶜ[i, j, 1]
@inline static_column_depthᶠᶜᵃ(i, j, grid::MultiEnvelopeGrid) = @inbounds grid.z.hᶠᶜ[i, j, 1]
@inline static_column_depthᶜᶠᵃ(i, j, grid::MultiEnvelopeGrid) = @inbounds grid.z.hᶜᶠ[i, j, 1]
@inline static_column_depthᶠᶠᵃ(i, j, grid::MultiEnvelopeGrid) = @inbounds grid.z.hᶠᶠ[i, j, 1]
