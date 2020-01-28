using Printf
using Oceananigans.Operators

using Oceananigans: AbstractGrid

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
`latitude` according to the relation `f = 2*rotation_rate*sind(latitude).

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
        return FPlane{FT}(2rotation_rate*sind(latitude))
    end
end

@inline x_f_cross_U(i, j, k, grid, coriolis::FPlane, U) = - coriolis.f * ℑxyᶠᶜᵃ(i, j, k, grid, U.v)
@inline y_f_cross_U(i, j, k, grid, coriolis::FPlane, U) =   coriolis.f * ℑxyᶜᶠᵃ(i, j, k, grid, U.u)
@inline z_f_cross_U(i, j, k, grid::AbstractGrid{FT}, coriolis::FPlane, U) where FT = zero(FT)

Base.show(io::IO, f_plane::FPlane{FT}) where FT =
    println(io, "FPlane{$FT}: f = ", @sprintf("%.2e", f_plane.f))


"""
    NonTraditionalFPlane{FT} <: AbstractRotation

A Coriolis implementation that facilitates non-traditional Coriolis terms in the zonal
and vertical momentum equations along with the traditional Coriolis terms.
"""
struct NonTraditionalFPlane{FT} <: AbstractRotation
    f  :: FT
    f′ :: FT
end

"""
    NonTraditionalFPlane([FT=Float64;] f=nothing, f′=nothing,
                                       rotation_rate=Ω_Earth, latitude=nothing)

Returns a parameter object for constant rotation about an arbitrary axis. The two
perpendicular components of rotation are given by angular frequencies `f/2` and `f′/2`
and therefore with background vorticities `f` and `f′`, respectively, which can be
directly specified.

In oceanography `f` and `f′` represent the components of planetary voriticity which
are perpendicular and parallel to the surface, respectively.

If `f` and `f′` are not specified, they are calculated from `rotation_rate` and `latitude`
according to the relations `f = 2*rotation_rate*sind(latitude)` and
`f′ = 2*rotation_rate*cosd(latitude)`, respectively. By default, `rotation_rate`
is assumed to be Earth's.
"""
function NonTraditionalFPlane(FT=Float64; f=nothing, f′=nothing, rotation_rate=Ω_Earth, latitude=nothing)

    use_f = !isnothing(f) && !isnothing(f′) && isnothing(latitude)
    use_planet_parameters = !isnothing(latitude) && isnothing(f) && isnothing(f′)

    if (use_planet_parameters && use_f) || (!use_planet_parameters && !use_f)
        throw(ArgumentError("Either both rotation_rate and latitude must be " *
                            "specified, *or* both f and f′ must be specified."))
    elseif use_planet_parameters
        f  = 2rotation_rate*sind(latitude)
        f′ = 2rotation_rate*cosd(latitude)
    end
    return NonTraditionalFPlane{FT}(f, f′)
end

@inline fv_minus_f′w(i, j, k, grid, coriolis::NonTraditionalFPlane, U) = coriolis.f′ * ℑzᵃᵃᶜ(i, j, k, grid, U.w) - coriolis.f * ℑyᵃᶜᵃ(i, j, k, grid, U.v)

@inline x_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) =   ℑxᶠᵃᵃ(i, j, k, grid, fv_minus_f′w, coriolis, U)
@inline y_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) =   coriolis.f  * ℑxyᶜᶠᵃ(i, j, k, grid, U.u)
@inline z_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) = - coriolis.f′ * ℑxzᶜᵃᶠ(i, j, k, grid, U.u)

Base.show(io::IO, f_plane::NonTraditionalFPlane{FT}) where FT =
    println(io, "Non-Traditional FPlane{$FT}:", @sprintf("f = %.2e, f′ = %.2e", f_plane.f, f_plane.f′))
