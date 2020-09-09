"""
    NonTraditionalBetaPlane{FT} <: AbstractRotation

A Coriolis implementation that accounts for the latitudinal variation of both
the locally vertical and the locally horizontal components of the rotation vector
of a sphere projected into a rectangular tangent plane.
The "traditional" approximation in ocean models accounts for only the locally
vertical component of the rotation vector (see [`BetaPlane`](@ref)).

This implementation follows section 5 of Dellar (2011) and conserves energy,
angular momentum, and potential vorticity.

References
==========
Dellar, P. (2011). Variations on a beta-plane: Derivation of non-traditional
    beta-plane equations from Hamilton's principle on a sphere. Journal of
    Fluid Mechanics, 674, 174-195. doi:10.1017/S0022112010006464
"""
struct NonTraditionalBetaPlane{FT} <: AbstractRotation
    fz :: FT
    fy :: FT
    βz :: FT
    βy :: FT
    R  :: FT
end

"""
    NonTraditionalBetaPlane(FT=Float64; fz=nothing, fy=nothing, βz=nothing, βy=nothing,
                                        rotation_rate=Ω_Earth, latitude=nothing, radius=R_Earth)

Returns an object representing the "non-traditional beta-plane" approximation
to the Coriolis force on a sphere. It's parameters are

    * `fz`: Projection of twice the planetary rotation rate on the local
            vertical (``z``) direction, evaluated at ``y = 0``.
    * `fy`: Projection of twice the planetary rotation rate on the local
            poleward (``y``) direction, evaluated at ``y = 0``.
    * `βz`: Projection of the latitudinal gradient of twice the planetary rotation
            rate on the local vertical (``z``) direction, evaluated at ``y = 0``.
    * `βy`: Projection of the latitudinal gradient of twice the planetary rotation
            rate on the local poleward (``y``) direction, evaluated at ``y = 0``.

The user may directly specify `fz`, `fy`, `βz`, `βy`, and `radius` or the three
parameters `rotation_rate`, `latitude`, and `radius` that specify the rotation rate
and radius of a planet, and the central latitude (where ``y = 0``) at which the
non-traditional `βz`-plane approximation is to be made.

The default `rotation_rate` and planet `radius` belong to Earth.
"""
function NonTraditionalBetaPlane(FT=Float64; fz=nothing, fy=nothing, βz=nothing, βy=nothing,
                                             rotation_rate=Ω_Earth, latitude=nothing, radius=R_Earth)

    Ω, φ, R = rotation_rate, latitude, radius

    use_f = !all(isnothing.((fz, fy, βz, βy))) && isnothing(latitude)
    use_planet_parameters = !isnothing(latitude) && all(isnothing.((fz, fy, βz, βy)))

    if !xor(use_f, use_planet_parameters)
        throw(ArgumentError("Either the keywords fz, fy, βz, βy, and radius must be specified, " *
                            "*or* all of rotation_rate, latitude, and radius."))
    end

    if use_planet_parameters
        fz =   2Ω*sind(φ)
        fy =   2Ω*cosd(φ)
        βz =   2Ω*cosd(φ)/R
        βy = - 4Ω*sind(φ)/R
    end

    return NonTraditionalBetaPlane{FT}(fz, fy, βz, βy, R)
end

@inline two_Ωʸ(P, y, z) = P.fy * (1 -  z/P.R) + P.βy * y
@inline two_Ωᶻ(P, y, z) = P.fz * (1 + 2z/P.R) + P.βz * y

# This function is eventually interpolated to fcc to contribute to x_f_cross_U.
@inline two_Ωʸw_minus_two_Ωᶻv(i, j, k, grid, coriolis, U) =
    @inbounds (  two_Ωʸ(coriolis, grid.yC[j], grid.zC[k]) * ℑzᵃᵃᶜ(i, j, k, grid, U.w)
               - two_Ωᶻ(coriolis, grid.yC[j], grid.zC[k]) * ℑyᵃᶜᵃ(i, j, k, grid, U.v))

@inline x_f_cross_U(i, j, k, grid, coriolis::NonTraditionalBetaPlane, U) =
    ℑxᶠᵃᵃ(i, j, k, grid, two_Ωʸw_minus_two_Ωᶻv, coriolis, U)

@inline y_f_cross_U(i, j, k, grid, coriolis::NonTraditionalBetaPlane, U) =
    @inbounds  two_Ωᶻ(coriolis, grid.yF[k], grid.zC[k]) * ℑxyᶜᶠᵃ(i, j, k, grid, U.u)

@inline z_f_cross_U(i, j, k, grid, coriolis::NonTraditionalBetaPlane, U) =
    @inbounds -two_Ωʸ(coriolis, grid.yC[j], grid.zF[k]) * ℑxzᶜᵃᶠ(i, j, k, grid, U.u)

Base.show(io::IO, β_plane::NonTraditionalBetaPlane{FT}) where FT =
    print(io, "NonTraditionalBetaPlane{$FT}: ",
          @sprintf("fz = %.2e, fy = %.2e, βz = %.2e, βy = %.2e, R = %.2e",
                   β_plane.fz, β_plane.fy, β_plane.βz, β_plane.βy, β_plane.R))
