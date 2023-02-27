"""
    struct BetaPlane{T} <: AbstractRotation

A parameter object for meridionally increasing Coriolis parameter (`f = f₀ + β y`)
that accounts for the variation of the locally vertical component of the rotation
vector with latitude.
"""
struct BetaPlane{T} <: AbstractRotation
    f₀ :: T
    β :: T
end

"""
    BetaPlane([T=Float64;] f₀=nothing, β=nothing,
                           rotation_rate=Ω_Earth, latitude=nothing, radius=R_Earth)

Return a ``β``-plane Coriolis parameter, ``f = f₀ + β y``. 

The user may specify both `f₀` and `β`, or the three parameters `rotation_rate`, `latitude`
(in degrees), and `radius` that specify the rotation rate and radius of a planet, and
the central latitude (where ``y = 0``) at which the `β`-plane approximation is to be made.

If `f₀` and `β` are not specified, they are calculated from `rotation_rate`, `latitude`,
and `radius` according to the relations `f₀ = 2 * rotation_rate * sind(latitude)` and
`β = 2 * rotation_rate * cosd(latitude) / radius`.

By default, the `rotation_rate` and planet `radius` are assumed to be Earth's.
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

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::BetaPlane) = coriolis.f₀ + coriolis.β * ynode(Face(), Face(), Center(), i, j, k, grid)

@inline x_f_cross_U(i, j, k, grid, coriolis::BetaPlane, U) =
    @inbounds - (coriolis.f₀ + coriolis.β * ynode(Face(), Center(), Center(), i, j, k, grid)) * ℑxyᶠᶜᵃ(i, j, k, grid, U[2])

@inline y_f_cross_U(i, j, k, grid, coriolis::BetaPlane, U) =
    @inbounds   (coriolis.f₀ + coriolis.β * ynode(Center(), Face(), Center(), i, j, k, grid)) * ℑxyᶜᶠᵃ(i, j, k, grid, U[1])

@inline z_f_cross_U(i, j, k, grid, coriolis::BetaPlane, U) = zero(grid)

function Base.summary(βplane::BetaPlane{FT}) where FT 
    fstr = prettysummary(βplane.f₀)
    βstr = prettysummary(βplane.β)
    return "BetaPlane{$FT}(f₀=$fstr, β=$βstr)"
end

Base.show(io::IO, βplane::BetaPlane) = print(io, summary(βplane))
