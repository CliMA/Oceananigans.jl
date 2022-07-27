using Oceananigans.Utils: prettysummary

"""
    FPlane{FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""
struct FPlane{FT} <: AbstractRotation
    f :: FT
end

"""
    FPlane([FT=Float64;] f=nothing, rotation_rate=Ω_Earth, latitude=nothing)

Returns a parameter object for constant rotation at the angular frequency
`f/2`, and therefore with background vorticity `f`, around a vertical axis.
If `f` is not specified, it is calculated from `rotation_rate` and
`latitude` (in degrees) according to the relation `f = 2 * rotation_rate * sind(latitude)`.

By default, `rotation_rate` is assumed to be Earth's.

Also called `FPlane`, after the "f-plane" approximation for the local effect of
a planet's rotation in a planar coordinate system tangent to the planet's surface.
"""
function FPlane(FT::DataType=Float64; f=nothing, rotation_rate=Ω_Earth, latitude=nothing)

    use_f = !isnothing(f)
    use_planet_parameters = !isnothing(latitude)

    if !xor(use_f, use_planet_parameters)
        throw(ArgumentError("Either both keywords rotation_rate and latitude must be " *
                            "specified, *or* only f must be specified."))
    end

    if use_f
        return FPlane{FT}(f)
    elseif use_planet_parameters
        return FPlane{FT}(2rotation_rate * sind(latitude))
    end
end

@inline x_f_cross_U(i, j, k, grid, coriolis::FPlane, U) = - coriolis.f * ℑxyᶠᶜᵃ(i, j, k, grid, U[2])
@inline y_f_cross_U(i, j, k, grid, coriolis::FPlane, U) =   coriolis.f * ℑxyᶜᶠᵃ(i, j, k, grid, U[1])
@inline z_f_cross_U(i, j, k, grid, coriolis::FPlane, U) = zero(eltype(grid))

function Base.summary(fplane::FPlane{FT}) where FT 
    fstr = prettysummary(fplane.f)
    return "FPlane{$FT}(f=$fstr)"
end

Base.show(io::IO, fplane::FPlane) = print(io, summary(fplane))

