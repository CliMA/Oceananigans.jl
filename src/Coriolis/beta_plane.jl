struct BetaPlane{S, FT} <: AbstractRotation{S}
    scheme :: S
    f₀ :: FT
    β :: FT
end

"""
    BetaPlane([FT=Float64;] f₀=nothing, β=nothing,
              scheme = EENConserving(),
              rotation_rate=Oceananigans.defaults.planet_rotation_rate,
              latitude=nothing, radius=Oceananigans.defaults.planet_radius)

Return a ``β``-plane Coriolis parameter, ``f = f₀ + β y`` with floating-point type `FT`.

The user may specify both `f₀` and `β`, or the three parameters `rotation_rate`, `latitude`
(in degrees), and `radius` that specify the rotation rate and radius of a planet, and
the central latitude (where ``y = 0``) at which the `β`-plane approximation is to be made.

If `f₀` and `β` are not specified, they are calculated from `rotation_rate`, `latitude`,
and `radius` according to the relations `f₀ = 2 * rotation_rate * sind(latitude)` and
`β = 2 * rotation_rate * cosd(latitude) / radius`.

By default, the `rotation_rate` and planet `radius` are assumed to be Earth's.
"""
function BetaPlane(FT=Oceananigans.defaults.FloatType;
                   scheme = EENConserving(),
                   f₀ = nothing,
                   β = nothing,
                   rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                   latitude = nothing,
                   radius = Oceananigans.defaults.planet_radius)

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

    f₀ = convert(FT, f₀)
    β = convert(FT, β)

    return BetaPlane{FT}(scheme, f₀, β)
end

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::BetaPlane) = coriolis.f₀ + coriolis.β * ynode(i, j, k, grid, face, face, center)
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::BetaPlane) = coriolis.f₀ + coriolis.β * ynode(i, j, k, grid, center, center, center)

function Base.summary(βplane::BetaPlane{FT}) where FT
    fstr = prettysummary(βplane.f₀)
    βstr = prettysummary(βplane.β)
    return "BetaPlane{$FT}(f₀=$fstr, β=$βstr)"
end

Base.show(io::IO, βplane::BetaPlane) = print(io, summary(βplane))
