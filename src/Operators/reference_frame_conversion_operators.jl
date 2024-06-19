# TODO: have a general Oceananigans-wide function that retrieves a pointwise
# value for a function, an array, a number, a field etc?
# This would be a generalization of `getbc` that could be used everywhere we need it
@inline getvalue(::Nothing,        i, j, k, grid, args...) = nothing
@inline getvalue(a::Number,        i, j, k, grid, args...) = a
@inline getvalue(a::AbstractArray, i, j, k, grid, args...) = @inbounds a[i, j, k]

"""
    intrinsic_vector(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ)

Convert the three-dimensional vector with components `uₑ, vₑ, wₑ` defined in an _extrinsic_ 
reference frame associated with the domain, to the reference frame _intrinsic_ to the grid. 

_extrinsic_ reference frames are:

- Cartesian for box domains 
- Latitude - Longitude for spherical domains

Therefore, for the `RectilinearGrid` and the `LatitudeLongitudeGrid`, the _extrinsic_ and the 
_intrinsic_ reference frames are equivalent
"""
@inline intrinsic_vector(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) = 
    getvalue(uₑ, i, j, k, grid), getvalue(vₑ, i, j, k, grid), getvalue(wₑ, i, j, k, grid)

"""
    extrinsic_vector(i, j, k, grid::AbstractGrid, uᵢ, vᵢ, wᵢ)

Convert the three-dimensional vector with components `uₑ, vₑ, wₑ` defined on the reference 
reference frame _intrinsic_ to the grid, to the _extrinsic_ reference frame associated with the domain.

_extrinsic_ reference frames are:

- Cartesian for box domains 
- Latitude - Longitude for spherical domains

Therefore, for the `RectilinearGrid` and the `LatitudeLongitudeGrid`, the _extrinsic_ and the 
_intrinsic_ reference frames are equivalent
"""
@inline extrinsic_vector(i, j, k, grid::AbstractGrid, uᵢ, vᵢ, wᵢ) =
    getvalue(uᵢ, i, j, k, grid), getvalue(vᵢ, i, j, k, grid), getvalue(wᵢ, i, j, k, grid)

@inline function extrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uₑ, vₑ, wₑ) 

    φᶜᶠᵃ₊ = φnode(i, j+1, 1, grid, Center(), Face(), Center())
    φᶜᶠᵃ₋ = φnode(i,   j, 1, grid, Center(), Face(), Center())
    Δyᶜᶜᵃ = Δyᶜᶜᶜ(i,   j, 1, grid)

    ũ = deg2rad(φᶜᶠᵃ₊ - φᶜᶠᵃ₋) / Δyᶜᶜᵃ

    φᶠᶜᵃ₊ = φnode(i+1, j, 1, grid, Face(), Center(), Center())
    φᶠᶜᵃ₋ = φnode(i,   j, 1, grid, Face(), Center(), Center())
    Δxᶜᶜᵃ = Δxᶜᶜᶜ(i,   j, 1, grid)

    ṽ = - deg2rad(φᶠᶜᵃ₊ - φᶠᶜᵃ₋) / Δxᶜᶜᵃ

    𝒰 = sqrt(ũ^2 + ṽ^2)

    u  = getvalue(uₑ, i, j, k, grid)
    v  = getvalue(vₑ, i, j, k, grid)
    wᵢ = getvalue(wₑ, i, j, k, grid)

    d₁ = ũ / 𝒰
    d₂ = ṽ / 𝒰

    uᵢ = u * d₁ - v * d₂
    vᵢ = u * d₂ + v * d₁

    return uᵢ, vᵢ, wᵢ
end

@inline function intrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uᵢ, vᵢ, wᵢ)

    φᶜᶠᵃ₊ = φnode(i, j+1, 1, grid, Center(), Face(), Center())
    φᶜᶠᵃ₋ = φnode(i,   j, 1, grid, Center(), Face(), Center())
    Δyᶜᶜᵃ = Δyᶜᶜᶜ(i,   j, 1, grid)

    ũ = deg2rad(φᶜᶠᵃ₊ - φᶜᶠᵃ₋) / Δyᶜᶜᵃ

    φᶠᶜᵃ₊ = φnode(i+1, j, 1, grid, Face(), Center(), Center())
    φᶠᶜᵃ₋ = φnode(i,   j, 1, grid, Face(), Center(), Center())
    Δxᶜᶜᵃ = Δxᶜᶜᶜ(i,   j, 1, grid)

    ṽ = - deg2rad(φᶠᶜᵃ₊ - φᶠᶜᵃ₋) / Δxᶜᶜᵃ

    𝒰 = sqrt(ũ^2 + ṽ^2)

    u  = getvalue(uᵢ, i, j, k, grid)
    v  = getvalue(vᵢ, i, j, k, grid)
    wₑ = getvalue(wᵢ, i, j, k, grid)

    d₁ = ũ / 𝒰
    d₂ = ṽ / 𝒰

    uₑ = u * d₁ + v * d₂
    vₑ = u * d₂ - v * d₁

    return uₑ, vₑ, wₑ
end

#####
##### Component-wise conversion between reference frames
#####

@inline intrinsic_vector_x_component(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) = 
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ, wₑ)[1]
    
@inline intrinsic_vector_y_component(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ, wₑ)[2]

@inline intrinsic_vector_z_component(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ, wₑ)[3]

@inline extrinsic_vector_x_component(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ, wₑ)[1]
    
@inline extrinsic_vector_y_component(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ, wₑ)[2]

@inline extrinsic_vector_z_component(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ, wₑ)[3]