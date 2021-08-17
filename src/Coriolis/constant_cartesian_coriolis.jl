using Oceananigans.Grids: ZDirection, validate_unit_vector

"""
    ConstantCartesianCoriolis{FT} <: AbstractRotation

A Coriolis implementation that accounts for the locally vertical and possibly both local horizontal
components of a constant rotation vector. A more general implementation of [`FPlane`](@ref), which only
accounts for the locally vertical component.
"""
struct ConstantCartesianCoriolis{FT} <: AbstractRotation
    fx :: FT
    fy :: FT
    fz :: FT
end

"""
    ConstantCartesianCoriolis([FT=Float64;] fx=nothing, fy=nothing, fz=nothing,
                                            f=nothing, rotation_axis=ZDirection(), 
                                            rotation_rate=Ω_Earth, latitude=nothing)

Returns a parameter object for a constant rotation decomposed into the `x`, `y` and `z` directions.
In oceanography the components `x`, `y`, `z` correspond to the directions east, north, and up. This
rotation can be specified in three different ways:

- Specifying all components `fx`, `fy` and `fz` directly.
- Specifying the Coriolis parameter `f` and (optionally) a `rotation_axis` (which defaults to the
  `z` direction if not specified).
- Specifying `latitude` (in degrees!) and (optionally) a `rotation_rate` in radians per second
  (which defaults to Earth's rotation rate).
"""
function ConstantCartesianCoriolis(FT=Float64; fx=nothing, fy=nothing, fz=nothing,
                                               f=nothing, rotation_axis=ZDirection(), 
                                               rotation_rate=Ω_Earth, latitude=nothing)
    if !isnothing(latitude)
        all(isnothing.((fx, fy, fz, f))) || throw(ArgumentError("Only `rotation_rate` can be specified when using `latitude`."))

        fx = 0
        fy = 2rotation_rate * cosd(latitude)
        fz = 2rotation_rate * sind(latitude)

    elseif !isnothing(f)
        all(isnothing.((fx, fy, fz, latitude))) || throw(ArgumentError("Only `rotation_axis` can be specified when using `f`."))

        rotation_axis = validate_unit_vector(rotation_axis)
        if rotation_axis isa ZDirection
            fx = fy = 0
            fz = f
        else
            fx = f * rotation_axis[1]
            fy = f * rotation_axis[2]
            fz = f * rotation_axis[3]
        end

    elseif all((!isnothing).((fx, fy, fz)))
        all(isnothing.((latitude, f))) || throw(ArgumentError("Only `fx`, `fy` and `fz` can be specified when setting ech component directly."))

    else
        throw(ArgumentError("At least `latitude`, or `f`, or `fx`, `fy` and `fz` must be specified."))
    end


    return ConstantCartesianCoriolis{FT}(fx, fy, fz)
end



# This function is eventually interpolated to fcc to contribute to x_f_cross_U.
@inline fʸw_minus_fᶻv(i, j, k, grid, coriolis, U) =
    coriolis.fy * ℑzᵃᵃᶜ(i, j, k, grid, U.w) - coriolis.fz * ℑyᵃᶜᵃ(i, j, k, grid, U.v)

@inline fᶻu_minus_fˣw(i, j, k, grid, coriolis, U) =
    coriolis.fz * ℑxᶜᵃᵃ(i, j, k, grid, U.u) - coriolis.fx * ℑzᵃᵃᶜ(i, j, k, grid, U.w)

@inline fˣv_minus_fʸu(i, j, k, grid, coriolis, U) =
    coriolis.fx * ℑyᵃᶜᵃ(i, j, k, grid, U.v) - coriolis.fy * ℑxᶜᵃᵃ(i, j, k, grid, U.u)

@inline x_f_cross_U(i, j, k, grid, coriolis::ConstantCartesianCoriolis, U) = ℑxᶠᵃᵃ(i, j, k, grid, fʸw_minus_fᶻv, coriolis, U)
@inline y_f_cross_U(i, j, k, grid, coriolis::ConstantCartesianCoriolis, U) = ℑyᵃᶠᵃ(i, j, k, grid, fᶻu_minus_fˣw, coriolis, U)
@inline z_f_cross_U(i, j, k, grid, coriolis::ConstantCartesianCoriolis, U) = ℑzᵃᵃᶠ(i, j, k, grid, fˣv_minus_fʸu, coriolis, U)

Base.show(io::IO, f_plane::ConstantCartesianCoriolis{FT}) where FT =
    print(io, "ConstantCartesianCoriolis{$FT}: ", @sprintf("fx = %.2e, fy = %.2e, fz = %.2e", f_plane.fx, f_plane.fy, f_plane.fz))
