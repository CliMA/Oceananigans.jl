"""
    BetaPlane{T} <: AbstractRotation

A parameter object for meridionally increasing Coriolis parameter (`f = f₀ + βy`).
"""
struct BetaPlane{T} <: AbstractRotation
    f₀ :: T
     β :: T
end

"""
    BetaPlane([T=Float64;] f₀=nothing, β=nothing,
                           rotation_rate=Ω_Earth, latitude=nothing, radius=R_Earth)

A parameter object for meridionally increasing Coriolis parameter (`f = f₀ + βy`).

The user may specify both `f₀` and `β`, or the three parameters `rotation_rate`,
`latitude`, and `radius` that specify the rotation rate and radius of a planet, and
the central latitude at which the `β`-plane approximation is to be made.

By default, the `rotation_rate` and planet `radius` is assumed to be Earth's.
"""
function BetaPlane(T=Float64; f₀=nothing, β=nothing,
                              rotation_rate=Ω_Earth, latitude=nothing, radius=R_Earth)

    use_f_and_β = !isnothing(f₀) && !isnothing(β)
    use_planet_parameters = !isnothing(latitude)

    if !xor(use_f_and_β, use_planet_parameters)
        throw(ArgumentError("Either both keywords f₀ and β must be specified, " *
                            "*or* all of rotation_rate, latitude, and radius."))
    end

    if use_planet_parameters
        f₀ = 2rotation_rate * sind(latitude)
         β = 2rotation_rate * cosd(latitude) / radius
     end

    return BetaPlane{T}(f₀, β)
end

@inline x_f_cross_U(i, j, k, grid, coriolis::BetaPlane, U) =
    @inbounds - (coriolis.f₀ + coriolis.β * grid.yC[j]) * ℑxyᶠᶜᵃ(i, j, k, grid, U.v)
@inline y_f_cross_U(i, j, k, grid, coriolis::BetaPlane, U) =
    @inbounds   (coriolis.f₀ + coriolis.β * grid.yF[j]) * ℑxyᶜᶠᵃ(i, j, k, grid, U.u)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, coriolis::BetaPlane, U) where FT = zero(FT)

Base.show(io::IO, β_plane::BetaPlane{FT}) where FT =
    println(io, "BetaPlane{$FT}: ", @sprintf("f₀ = %.2e, β = %.2e", β_plane.f₀, β_plane.β))
