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
    fz :: FT
    fy :: FT
end

"""
    NonTraditionalFPlane([FT=Float64;] fz=nothing, fy=nothing,
                                       rotation_rate=Ω_Earth, latitude=nothing)

Returns a parameter object for constant rotation about an axis in the `y-z` plane
with `y`- and `z`-components `fy/2` and `fz/2`, and the background vorticity is
`(0, fy, fz)`.

In oceanography `fz` and `fy` represent the components of planetary voriticity which
are perpendicular and parallel to the ocean surface in a domain in which `x, y, z` 
correspond to the directions east, north, and up.

If `fz` and `fy` are not specified, they are calculated from `rotation_rate` and `latitude`
according to the relations `fz = 2*rotation_rate*sind(latitude)` and
`fy = 2*rotation_rate*cosd(latitude)`, respectively. By default, `rotation_rate`
is assumed to be Earth's.
"""
function NonTraditionalFPlane(FT=Float64; fz=nothing, fy=nothing, rotation_rate=Ω_Earth, latitude=nothing)

    use_f = !isnothing(fz) && !isnothing(fy) && isnothing(latitude)
    use_planet_parameters = !isnothing(latitude) && isnothing(fz) && isnothing(fy)

    if (use_planet_parameters && use_f) || (!use_planet_parameters && !use_f)
        throw(ArgumentError("Either both rotation_rate and latitude must be " *
                            "specified, *or* both f and f′ must be specified."))
    elseif use_planet_parameters
        fz = 2rotation_rate*sind(latitude)
        fy = 2rotation_rate*cosd(latitude)
    end
    return NonTraditionalFPlane{FT}(fz, fy)
end

@inline fᶻv_minus_fʸw(i, j, k, grid, coriolis::NonTraditionalFPlane, U) = coriolis.fy * ℑzᵃᵃᶜ(i, j, k, grid, U.w) - coriolis.fz * ℑyᵃᶜᵃ(i, j, k, grid, U.v)

@inline x_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) =   ℑxᶠᵃᵃ(i, j, k, grid, fᶻv_minus_fʸw, coriolis, U)
@inline y_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) =   coriolis.fz * ℑxyᶜᶠᵃ(i, j, k, grid, U.u)
@inline z_f_cross_U(i, j, k, grid, coriolis::NonTraditionalFPlane, U) = - coriolis.fy * ℑxzᶜᵃᶠ(i, j, k, grid, U.u)

Base.show(io::IO, f_plane::NonTraditionalFPlane{FT}) where FT =
    println(io, "Non-Traditional FPlane{$FT}:", @sprintf("fz = %.2e, fy = %.2e", f_plane.fz, f_plane.fy))
