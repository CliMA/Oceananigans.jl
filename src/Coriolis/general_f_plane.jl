using Oceananigans.Grids: ZDirection, validate_unit_vector

"""
    GeneralFPlane{FT} <: AbstractRotation

A Coriolis implementation that accounts for both the locally vertical and
the locally horizontal components of the rotation vector. Traditionally
(see [`FPlane`](@ref)) only the locally vertical component is accounted for.
"""
struct GeneralFPlane{FT} <: AbstractRotation
    fx :: FT
    fy :: FT
    fz :: FT
end

"""
    GeneralFPlane([FT=Float64;] coriolis_frequency=Ω_Earth, rotation_axis=ZDirection())

Returns a parameter object for constant rotation about an axis in the `y-z` plane
with `y`- and `z`-components `fy/2` and `fz/2`, and the background vorticity is
`(0, fy, fz)`.

In oceanography `fz` and `fy` represent the components of planetary voriticity which
are perpendicular and parallel to the ocean surface in a domain in which `x, y, z`
correspond to the directions east, north, and up.

If `fz` and `fy` are not specified, they are calculated from `coriolis_frequency` and `latitude`
according to the relations `fz = 2 * coriolis_frequency * sind(latitude)` and
`fy = 2 * coriolis_frequency * cosd(latitude)`, respectively. By default, `coriolis_frequency`
is assumed to be Earth's.
"""
function GeneralFPlane(FT=Float64; coriolis_frequency=2Ω_Earth, rotation_axis=ZDirection())
    rotation_axis = validate_unit_vector(rotation_axis)

    if rotation_axis isa ZDirection
        fx = fy = 0
        fz = coriolis_frequency
    else
        fx = coriolis_frequency * rotation_axis[1]
        fy = coriolis_frequency * rotation_axis[2]
        fz = coriolis_frequency * rotation_axis[3]
    end

    return GeneralFPlane{FT}(fx, fy, fz)
end



# This function is eventually interpolated to fcc to contribute to x_f_cross_U.
@inline fʸw_minus_fᶻv(i, j, k, grid, coriolis, U) =
    coriolis.fy * ℑzᵃᵃᶜ(i, j, k, grid, U.w) - coriolis.fz * ℑyᵃᶜᵃ(i, j, k, grid, U.v)

@inline fᶻu_minus_fˣw(i, j, k, grid, coriolis, U) =
    coriolis.fz * ℑxᶜᵃᵃ(i, j, k, grid, U.u) - coriolis.fx * ℑzᵃᵃᶜ(i, j, k, grid, U.w)

@inline fˣv_minus_fʸu(i, j, k, grid, coriolis, U) =
    coriolis.fx * ℑyᵃᶜᵃ(i, j, k, grid, U.v) - coriolis.fy * ℑxᶜᵃᵃ(i, j, k, grid, U.u)

@inline x_f_cross_U(i, j, k, grid, coriolis::GeneralFPlane, U) = ℑxᶠᵃᵃ(i, j, k, grid, fʸw_minus_fᶻv, coriolis, U)
@inline y_f_cross_U(i, j, k, grid, coriolis::GeneralFPlane, U) = ℑyᵃᶠᵃ(i, j, k, grid, fᶻu_minus_fˣw, coriolis, U)
@inline z_f_cross_U(i, j, k, grid, coriolis::GeneralFPlane, U) = ℑzᵃᵃᶠ(i, j, k, grid, fˣv_minus_fʸu, coriolis, U)

Base.show(io::IO, f_plane::GeneralFPlane{FT}) where FT =
    print(io, "GeneralFPlane{$FT}: ", @sprintf("fx = %.2e, fy = %.2e, fz = %.2e", f_plane.fx, f_plane.fy, f_plane.fz))
