#####
##### ZStar coordinate and associated types
#####

abstract type AbstractVerticalCoordinate end

#####
##### AbstractVerticalCoordinate grid definitions
#####

const AVLLG  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractVerticalCoordinate}
const AVOSSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractVerticalCoordinate}
const AVRG   = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractVerticalCoordinate}

const AbstractVerticalCoordinateUnderlyingGrid = Union{AVLLG, AVOSSG, AVRG}

# rnode for an AbstractVerticalCoordinate grid is the reference node
@inline rnode(i, j, k, grid::AbstractVerticalCoordinateUnderlyingGrid, ℓx, ℓy, ::Center) = @inbounds grid.zᵃᵃᶜ.reference[k] 
@inline rnode(i, j, k, grid::AbstractVerticalCoordinateUnderlyingGrid, ℓx, ℓy, ::Face)   = @inbounds grid.zᵃᵃᶠ.reference[k] 

function retrieve_static_grid(grid::AbstractVerticalCoordinateUnderlyingGrid) 

    Δzᵃᵃᶠ = reference_zspacings(grid, Face())
    Δzᵃᵃᶜ = reference_zspacings(grid, Center())

    TX, TY, TZ = topology(grid)

    args = []
    for prop in propertynames(grid)
        if prop == :Δzᵃᵃᶠ
            push!(args, Δzᵃᵃᶠ)
        elseif prop == :Δzᵃᵃᶜ
            push!(args, Δzᵃᵃᶜ)
        else
            push!(args, getproperty(grid, prop))
        end
    end

    GridType = getnamewrapper(grid)

    return GridType{TX, TY, TZ}(args...)
end

"""
    struct ZStarVerticalCoordinate{R, S} <: AbstractVerticalSpacing{R}

A vertical coordinate for the hydrostatic free surface model that follows the free surface.
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
struct ZStarVerticalCoordinate{R, SCC, SFC, SCF, SFF} <: AbstractVerticalCoordinate
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

generate_coordinate(FT, ::Periodic, N, H, ::ZStarVerticalCoordinate, coordinate_name, arch, args...) = 
    throw(ArgumentError("Periodic domains are not supported for ZStarVerticalCoordinate"))

# Generate a regularly-spaced coordinate passing the domain extent (2-tuple) and number of points
function generate_coordinate(FT, topo, size, halo, coordinate::ZStarVerticalCoordinate, coordinate_name, dim::Int, arch)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    if dim != 3 
        msg = "ZStarVerticalCoordinate is supported only in the third dimension (z)"
        throw(ArgumentError(msg))
    end

    if coordinate_name != :z
        msg = "Only z-coordinate is supported for ZStarVerticalCoordinate"
        throw(ArgumentError(msg))
    end

    r_faces = coordinate.reference

    Lr, rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ = generate_coordinate(FT, topo[3](), Nz, Hz, r_faces, :z, arch)

    args = (topo, (Nx, Ny, Nz), (Hx, Hy, Hz))

    sᶜᶜᵃ = new_data(FT, arch, (Center, Center, Nothing), args...)
    sᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    sᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), args...)
    sᶠᶠᵃ = new_data(FT, arch, (Face,   Face,   Nothing), args...)  

    sᶜᶜᵃ₋ = new_data(FT, arch, (Center, Center, Nothing), args...)
    sᶜᶠᵃ₋ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    sᶠᶜᵃ₋ = new_data(FT, arch, (Face,   Center, Nothing), args...)

    ∂t_s = new_data(FT, arch, (Center, Center, Nothing), args...)
    # Storage place for the free surface height? Probably find a better way to call this
    η    = new_data(FT, arch, (Center, Center, Nothing), args...)

    # fill all the scalings with 1
    for s in (sᶜᶜᵃ, sᶜᶠᵃ, sᶠᶜᵃ, sᶠᶠᵃ, sᶜᶜᵃ₋, sᶜᶠᵃ₋, sᶠᶜᵃ₋)
        fill!(s, 1)
    end

    # The scaling is the same for everyone (H + \eta) / H, the vertical coordinate requires 
    # to add the free surface to retrieve the znode.
    zᵃᵃᶠ = ZStarVerticalCoordinate(rᵃᵃᶠ, sᶜᶜᵃ, sᶠᶜᵃ, sᶜᶠᵃ, sᶠᶠᵃ, sᶜᶜᵃ₋, sᶠᶜᵃ₋, sᶜᶠᵃ₋, η)
    zᵃᵃᶜ = ZStarVerticalCoordinate(rᵃᵃᶜ, sᶜᶜᵃ, sᶠᶜᵃ, sᶜᶠᵃ, sᶠᶠᵃ, sᶜᶜᵃ₋, sᶠᶜᵃ₋, sᶜᶠᵃ₋, η)

    Δzᵃᵃᶠ = ZStarVerticalCoordinate(Δrᵃᵃᶠ, sᶜᶜᵃ, sᶠᶜᵃ, sᶜᶠᵃ, sᶠᶠᵃ, sᶜᶜᵃ₋, sᶠᶜᵃ₋, sᶜᶠᵃ₋, ∂t_s)
    Δzᵃᵃᶜ = ZStarVerticalCoordinate(Δrᵃᵃᶜ, sᶜᶜᵃ, sᶠᶜᵃ, sᶜᶠᵃ, sᶠᶠᵃ, sᶜᶜᵃ₋, sᶠᶜᵃ₋, sᶜᶠᵃ₋, ∂t_s)

    return Lr, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
end

function validate_dimension_specification(T, ξ::ZStarVerticalCoordinate, dir, N, FT)
    reference = validate_dimension_specification(T, ξ.reference, dir, N, FT)
    args      = Tuple(getproperty(ξ, prop) for prop in propertynames(ξ))

    return ZStarVerticalCoordinate(reference, args[2:end]...)
end

const ZStarLLG  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const ZStarOSSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const ZStarRG   = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}

const ZStarUnderlyingGrid = Union{ZStarLLG, ZStarOSSG, ZStarRG}

#####
##### ZStar-specific vertical spacing functions
#####

const C = Center
const F = Face

const ZSG = ZStarUnderlyingGrid

# Fallbacks
reference_zspacings(grid, ::C) = grid.Δzᵃᵃᶜ
reference_zspacings(grid, ::F) = grid.Δzᵃᵃᶠ

@inline vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline previous_vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline ∂t_grid(i, j, k, grid) = zero(grid)
@inline V_times_∂t_grid(i, j, k, grid) = zero(grid)

reference_zspacings(grid::ZSG, ::C) = grid.Δzᵃᵃᶜ.reference
reference_zspacings(grid::ZSG, ::F) = grid.Δzᵃᵃᶠ.reference

@inline vertical_scaling(i, j, k, grid::ZSG, ::C, ::C, ::C) = @inbounds grid.Δzᵃᵃᶜ.sᶜᶜⁿ[i, j]
@inline vertical_scaling(i, j, k, grid::ZSG, ::F, ::C, ::C) = @inbounds grid.Δzᵃᵃᶜ.sᶠᶜⁿ[i, j]
@inline vertical_scaling(i, j, k, grid::ZSG, ::C, ::F, ::C) = @inbounds grid.Δzᵃᵃᶜ.sᶜᶠⁿ[i, j]
@inline vertical_scaling(i, j, k, grid::ZSG, ::F, ::F, ::C) = @inbounds grid.Δzᵃᵃᶜ.sᶠᶠⁿ[i, j]

@inline vertical_scaling(i, j, k, grid::ZSG, ::C, ::C, ::F) = @inbounds grid.Δzᵃᵃᶠ.sᶜᶜⁿ[i, j]
@inline vertical_scaling(i, j, k, grid::ZSG, ::F, ::C, ::F) = @inbounds grid.Δzᵃᵃᶠ.sᶠᶜⁿ[i, j]
@inline vertical_scaling(i, j, k, grid::ZSG, ::C, ::F, ::F) = @inbounds grid.Δzᵃᵃᶠ.sᶜᶠⁿ[i, j]
@inline vertical_scaling(i, j, k, grid::ZSG, ::F, ::F, ::F) = @inbounds grid.Δzᵃᵃᶠ.sᶠᶠⁿ[i, j]

@inline previous_vertical_scaling(i, j, k, grid::ZSG, ::C, ::C, ::C) = @inbounds grid.Δzᵃᵃᶜ.sᶜᶜ⁻[i, j]
@inline previous_vertical_scaling(i, j, k, grid::ZSG, ::F, ::C, ::C) = @inbounds grid.Δzᵃᵃᶜ.sᶠᶜ⁻[i, j]
@inline previous_vertical_scaling(i, j, k, grid::ZSG, ::C, ::F, ::C) = @inbounds grid.Δzᵃᵃᶜ.sᶜᶠ⁻[i, j]

@inline previous_vertical_scaling(i, j, k, grid::ZSG, ::C, ::C, ::F) = @inbounds grid.Δzᵃᵃᶠ.sᶜᶜ⁻[i, j]
@inline previous_vertical_scaling(i, j, k, grid::ZSG, ::F, ::C, ::F) = @inbounds grid.Δzᵃᵃᶠ.sᶠᶜ⁻[i, j]
@inline previous_vertical_scaling(i, j, k, grid::ZSG, ::C, ::F, ::F) = @inbounds grid.Δzᵃᵃᶠ.sᶜᶠ⁻[i, j]

@inline ∂t_grid(i, j, k, grid::ZSG) = @inbounds grid.Δzᵃᵃᶜ.∂t_s[i, j] 
@inline V_times_∂t_grid(i, j, k, grid::ZSG) = ∂t_grid(i, j, k, grid) * Vᶜᶜᶜ(i, j, k, grid)

#####
##### znode
#####

const c = Center()
const f = Face()

# rnode for an AbstractVerticalCoordinate grid is the reference node
@inline znode(i, j, k, grid::ZSG, ::C, ::C, ::C) = @inbounds grid.zᵃᵃᶜ.reference[k] * vertical_scaling(i, j, k, grid, c, c, c) + grid.zᵃᵃᶜ.∂t_s[i, j] 
@inline znode(i, j, k, grid::ZSG, ::C, ::F, ::C) = @inbounds grid.zᵃᵃᶜ.reference[k] * vertical_scaling(i, j, k, grid, c, f, c) + grid.zᵃᵃᶜ.∂t_s[i, j] 
@inline znode(i, j, k, grid::ZSG, ::F, ::C, ::C) = @inbounds grid.zᵃᵃᶜ.reference[k] * vertical_scaling(i, j, k, grid, f, c, c) + grid.zᵃᵃᶜ.∂t_s[i, j] 
@inline znode(i, j, k, grid::ZSG, ::F, ::F, ::C) = @inbounds grid.zᵃᵃᶜ.reference[k] * vertical_scaling(i, j, k, grid, f, f, c) + grid.zᵃᵃᶜ.∂t_s[i, j] 
@inline znode(i, j, k, grid::ZSG, ::C, ::C, ::F) = @inbounds grid.zᵃᵃᶠ.reference[k] * vertical_scaling(i, j, k, grid, c, c, c) + grid.zᵃᵃᶠ.∂t_s[i, j] 
@inline znode(i, j, k, grid::ZSG, ::C, ::F, ::F) = @inbounds grid.zᵃᵃᶠ.reference[k] * vertical_scaling(i, j, k, grid, c, f, c) + grid.zᵃᵃᶠ.∂t_s[i, j] 
@inline znode(i, j, k, grid::ZSG, ::F, ::C, ::F) = @inbounds grid.zᵃᵃᶠ.reference[k] * vertical_scaling(i, j, k, grid, f, c, c) + grid.zᵃᵃᶠ.∂t_s[i, j] 
@inline znode(i, j, k, grid::ZSG, ::F, ::F, ::F) = @inbounds grid.zᵃᵃᶠ.reference[k] * vertical_scaling(i, j, k, grid, f, f, c) + grid.zᵃᵃᶠ.∂t_s[i, j] 