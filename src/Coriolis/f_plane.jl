using Oceananigans.Utils: prettysummary

"""
    struct FPlane{S, FT} <: AbstractRotation{S}

A parameter object for constant rotation around a vertical axis.
"""
struct FPlane{S, FT} <: AbstractRotation{S}
    scheme :: S
    f :: FT
end

"""
    FPlane([FT = Oceananigans.defaults.FloatType;]
           f = nothing,
           scheme = EENConserving(),
           rotation_rate = Oceananigans.defaults.planet_rotation_rate,
           latitude = nothing)

Return a parameter object for constant rotation at the angular frequency
`f/2`, and therefore with background vorticity `f`, around a vertical axis.
If `f` is not specified, it is calculated from `rotation_rate` and
`latitude` (in degrees) according to the relation `f = 2 * rotation_rate * sind(latitude)`.

By default, `rotation_rate` is assumed to be Earth's.

Also called `FPlane`, after the "f-plane" approximation for the local effect of
a planet's rotation in a planar coordinate system tangent to the planet's surface.
"""
function FPlane(FT::DataType=Oceananigans.defaults.FloatType;
                f = nothing,
                scheme = EENConserving(),
                rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                latitude = nothing)

    use_f = !isnothing(f)
    use_planet_parameters = !isnothing(latitude)

    if !xor(use_f, use_planet_parameters)
        throw(ArgumentError("Either both keywords rotation_rate and latitude must be " *
                            "specified, *or* only f must be specified."))
    end

    if use_f
        f = convert(FT, f)
    elseif use_planet_parameters
        f = convert(FT, 2rotation_rate * sind(latitude))
    end

    return FPlane(scheme, f)
end

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::FPlane) = coriolis.f
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::FPlane) = coriolis.f

function Base.summary(fplane::FPlane{FT}) where FT
    fstr = prettysummary(fplane.f)
    return "FPlane{$FT}(f=$fstr)"
end

Base.show(io::IO, fplane::FPlane) = print(io, summary(fplane))
