
#####
##### ZStar coordinate and associated grid types
#####

"""
    struct ZStarSpacing{R, S} <: AbstractVerticalSpacing{R}

A vertical spacing for the hydrostatic free surface model that follows the free surface.
The vertical spacing is defined by a reference spacing `Δr` and a scaling `s` that obeys
```math
s = (η + H) / H
```
where ``η`` is the free surface height and ``H`` the vertical depth of the water column

# Fields
- `Δr`: reference vertical spacing with `η = 0`
- `sᶜᶜⁿ`: scaling of the vertical coordinate at time step `n` at `(Center, Center, Any)` location
- `sᶠᶜⁿ`: scaling of the vertical coordinate at time step `n` at `(Face,   Center, Any)` location
- `sᶜᶠⁿ`: scaling of the vertical coordinate at time step `n` at `(Center, Face,   Any)` location
- `sᶠᶠⁿ`: scaling of the vertical coordinate at time step `n` at `(Face,   Face,   Any)` location
- `s⁻`: scaling of the vertical coordinate at time step `n - 1` at `(Center, Center, Any)` location
- `∂t_s`: Time derivative of `s`
"""
struct ZStarVerticalCoordinate{R, SCC, SFC, SCF, SFF} <: AbstractVerticalCoordinate{R}
    reference :: R
         sᶜᶜⁿ :: SCC
         sᶠᶜⁿ :: SFC
         sᶜᶠⁿ :: SCF
         sᶠᶠⁿ :: SFF
         sᶜᶜ⁻ :: SCC
         sᶠᶜ⁻ :: SFC
         sᶜᶠ⁻ :: SCF
        ∂t_s  :: SCC
end

ZStarVerticalCoordinate(r_faces) = ZStarVerticalCoordinate(r_faces, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, coord::ZStarVerticalCoordinate) = 
            ZStarSpacing(Adapt.adapt(to, coord.reference),
                         Adapt.adapt(to, coord.sᶜᶜⁿ),
                         Adapt.adapt(to, coord.sᶠᶜⁿ),
                         Adapt.adapt(to, coord.sᶜᶠⁿ),
                         Adapt.adapt(to, coord.sᶠᶠⁿ),
                         Adapt.adapt(to, coord.sᶜᶜ⁻),
                         Adapt.adapt(to, coord.sᶠᶜ⁻),
                         Adapt.adapt(to, coord.sᶜᶠ⁻),
                         Adapt.adapt(to, coord.∂t_s))

on_architecture(arch, coord::ZStarVerticalCoordinate) = 
            ZStarSpacing(on_architecture(arch, coord.reference),
                         on_architecture(arch, coord.sᶜᶜⁿ),
                         on_architecture(arch, coord.sᶠᶜⁿ),
                         on_architecture(arch, coord.sᶜᶠⁿ),
                         on_architecture(arch, coord.sᶠᶠⁿ),
                         on_architecture(arch, coord.sᶜᶜ⁻),
                         on_architecture(arch, coord.sᶠᶜ⁻),
                         on_architecture(arch, coord.sᶜᶠ⁻),
                         on_architecture(arch, coord.∂t_s))

Grids.coordinate_summary(::Bounded, Δ::ZStarVerticalCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(Δ.reference))


generate_coordinate(FT, topo::Periodic, N, H, coordinate::ZStarVerticalCoordinate, coordinate_name, arch, args...) = 
    throw(ArgumentError("Periodic domains are not supported for ZStarVerticalCoordinate"))

# Generate a regularly-spaced coordinate passing the domain extent (2-tuple) and number of points
function generate_coordinate(FT, topo::Bounded, Nz, Hz, coordinate::ZStarVerticalCoordinate, coordinate_name, arch, Nx, Ny, Hx, Hy)

    if coordinate_name != :z
        msg = "Only z-coordinate is supported for ZStarVerticalCoordinate"
        throw(ArgumentError(msg))
    end

    r_faces = coordinate.reference

    Lr, rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ = generate_coordinate(FT, topo[3](), Nz, Hz, r_faces, :z, arch)

    args = (topo, (Nx, Ny, Nz), (Hx, Hy, Hz))

    szᶜᶜᵃ = new_data(FT, arch, (Center, Center, Nothing), args...)
    szᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    szᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), args...)
    szᶠᶠᵃ = new_data(FT, arch, (Face,   Face,   Nothing), args...)
    
    sΔzᶜᶜᵃ = new_data(FT, arch, (Center, Center, Nothing), args...)
    sΔzᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    sΔzᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), args...)
    sΔzᶠᶠᵃ = new_data(FT, arch, (Face,   Face,   Nothing), args...)  

    sΔzᶜᶜᵃ₋ = new_data(FT, arch, (Center, Center, Nothing), args...)
    sΔzᶜᶠᵃ₋ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    sΔzᶠᶜᵃ₋ = new_data(FT, arch, (Face,   Center, Nothing), args...)

    ∂t_s = new_data(FT, arch, (Center, Center, Nothing), args...)

    zᵃᵃᶠ = ZStarVerticalCoordinate(rᵃᵃᶠ, szᶜᶜᵃ, szᶠᶜᵃ, szᶜᶠᵃ, szᶠᶠᵃ, nothing, nothing, nothing, nothing)
    zᵃᵃᶜ = ZStarVerticalCoordinate(rᵃᵃᶜ, szᶜᶜᵃ, szᶠᶜᵃ, szᶜᶠᵃ, szᶠᶠᵃ, nothing, nothing, nothing, nothing)

    Δzᵃᵃᶠ = ZStarVerticalCoordinate(Δrᵃᵃᶠ, sΔzᶜᶜᵃ, sΔzᶠᶜᵃ, sΔzᶜᶠᵃ, sΔzᶠᶠᵃ, sΔzᶜᶜᵃ₋, sΔzᶠᶜᵃ₋, sΔzᶜᶠᵃ₋, ∂t_s)
    Δzᵃᵃᶜ = ZStarVerticalCoordinate(Δrᵃᵃᶜ, sΔzᶜᶜᵃ, sΔzᶠᶜᵃ, sΔzᶜᶠᵃ, sΔzᶠᶠᵃ, sΔzᶜᶜᵃ₋, sΔzᶠᶜᵃ₋, sΔzᶜᶠᵃ₋, ∂t_s)

    return Lr, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
end

