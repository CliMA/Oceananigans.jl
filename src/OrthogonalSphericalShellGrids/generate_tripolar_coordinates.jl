using CubedSphere.SphericalGeometry: lat_lon_to_cartesian, spherical_area_quadrilateral

"""
    _compute_tripolar_coordinates!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC,
                                   λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                   first_pole_longitude,
                                   focal_distance, Nλ)

Compute the tripolar coordinates for a given set of input parameters following
the formulation by [Murray (1996)](@cite Murray1996).

The tripolar grid is built as a set of cofocal ellipses and perpendicular hyperbolae.
The `focal_distance` argument is the distance from the center of the ellipses to the foci.

The family of ellipses obeys:

```math
\\frac{x²}{a² \\cosh²(ψ)} + \\frac{y²}{a² \\sinh²(ψ)} = 1
```

While the family of perpendicular hyperbolae obey:

```math
\\frac{x²}{a² \\cosh²(λ)} + \\frac{y²}{a² \\sinh²(λ)} = 1
```

Where ``a`` is the `focal_distance` to the center, ``λ`` is the longitudinal angle,
and ``ψ`` is the "isometric latitude", defined by Murray (1996) and satisfying:

```math
    a \\sinh(ψ) = \\mathrm{tand}[(90 - φ) / 2]
```

The final ``(x, y)`` points that define the stereographic projection of the tripolar
coordinates are given by:

```math
    \\begin{align}
    x & = a \\sinh ψ \\cos λ \\\\
    y & = a \\sinh ψ \\sin λ
    \\end{align}
```

for which it is possible to retrieve the longitude and latitude by:

```math
    \\begin{align}
    λ &=    - \\frac{180}{π} \\mathrm{atan}(y / x)  \\\\
    φ &= 90 - \\frac{360}{π} \\mathrm{atan} \\sqrt{x² + y²}
    \\end{align}

References
==========

Murray, R. J. (1996). Explicit generation of orthogonal grids for ocean models.
    Journal of Computational Physics, 126(2), 251-273.
```
"""
@kernel function _compute_tripolar_coordinates!(λFA, φFA, λCA, φCA,
                                                  λF, λC, φA,
                                                  first_pole_longitude,
                                                  focal_distance, Nx)

    i, j = @index(Global, NTuple)

    λ2Ds = (λFA, λCA)
    φ2Ds = (φFA, φCA)
    λ1Ds = (λF , λC )
    φ1Ds = (φA , φA )
    iscenters = (false, true)

    for (λ2D, φ2D, λ1D, φ1D, iscenter) in zip(λ2Ds, φ2Ds, λ1Ds, φ1Ds, iscenters)
        # We chose the formulae below for λ ∈ (-180, 180) and φ ∈ (-90, 90)
        # so that the grid of (x,y) never crosses the negative x-axis,
        # overwhich atan is discontinuous, which we want to avoid.
        ψ = asinh(tand((90 - φ1D[j]) / 2) / focal_distance)
        x = focal_distance * cosd(λ1D[i]) * sinh(ψ)
        y = focal_distance * sind(λ1D[i]) * cosh(ψ)
        R = sqrt(x^2 + y^2)

        # λ is simply
        λ2D[i, j] = atand(y, x)
        # But we fill the halos ourselves here.
        # Instead of periodicity, we continue the longitudes in the East and West halos.
        λ2D[i, j] = ifelse(i < 1 && 1 ≤ j, λ2D[i, j] - 360, λ2D[i, j])
        # For the East halo, we need to "specialise" on center/face x-location.
        imax = ifelse(iscenter, Nx, Nx + 1)
        λ2D[i, j] = ifelse(imax < i && 1 ≤ j, λ2D[i, j] + 360, λ2D[i, j])
        # And we "continue" down South for j < 1 in the south halo.
        λ2D[i, j] = ifelse(j < 1, λ1D[i], λ2D[i, j])

        # And φ is simply
        φ2D[i, j] = 2 * atand(1, R) - 90
        # But we "continue" South until we hit the South Polefor j < 1 in the south halo.
        φ2D[i, j] = ifelse(j < 1, max(φ1D[j], -90), φ2D[i, j])

        # Make sure the singularities are at longitude we want them to be at.
        # (`first_pole_longitude` and `first_pole_longitude` + 180)
        λ2D[i, j] += first_pole_longitude + 90
    end
end

# Calculate the metric terms from the coordinates of the grid
# Note: There is probably a better way to do this, in Murray (2016) they give analytical
# expressions for the metric terms.
@kernel function _calculate_metrics!(Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                     Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                     Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                     λᶠᶜᵃ, λᶜᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                     φᶠᶜᵃ, φᶜᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, radius)

    i, j = @index(Global, NTuple)

    @inbounds begin
        Δxᶜᶜᵃ[i, j] = haversine((λᶠᶜᵃ[i+1, j], φᶠᶜᵃ[i+1, j]), (λᶠᶜᵃ[i, j],   φᶠᶜᵃ[i, j]),   radius)
        Δxᶠᶜᵃ[i, j] = haversine((λᶜᶜᵃ[i, j],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i-1, j], φᶜᶜᵃ[i-1, j]), radius)
        Δxᶜᶠᵃ[i, j] = haversine((λᶠᶠᵃ[i+1, j], φᶠᶠᵃ[i+1, j]), (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius)
        Δxᶠᶠᵃ[i, j] = haversine((λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   (λᶜᶠᵃ[i-1, j], φᶜᶠᵃ[i-1, j]), radius)

        Δyᶜᶜᵃ[i, j] = haversine((λᶜᶠᵃ[i, j+1], φᶜᶠᵃ[i, j+1]),   (λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   radius)
        Δyᶠᶜᵃ[i, j] = haversine((λᶠᶠᵃ[i, j+1], φᶠᶠᵃ[i, j+1]),   (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius)
        Δyᶜᶠᵃ[i, j] = haversine((λᶜᶜᵃ[i, j  ],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i, j-1], φᶜᶜᵃ[i, j-1]), radius)
        Δyᶠᶠᵃ[i, j] = haversine((λᶠᶜᵃ[i, j  ],   φᶠᶜᵃ[i, j]),   (λᶠᶜᵃ[i, j-1], φᶠᶜᵃ[i, j-1]), radius)

        a = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ])
        b = lat_lon_to_cartesian(φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ])
        c = lat_lon_to_cartesian(φᶠᶠᵃ[i+1, j+1], λᶠᶠᵃ[i+1, j+1])
        d = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1])

        Azᶜᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2

        # To be able to conserve kinetic energy specifically the momentum equation,
        # it is better to define the face areas as products of
        # the edge lengths rather than using the spherical area of the face (cit JMC).
        # TODO: find a reference to support this statement
        Azᶠᶜᵃ[i, j] = Δyᶠᶜᵃ[i, j] * Δxᶠᶜᵃ[i, j]
        Azᶜᶠᵃ[i, j] = Δyᶜᶠᵃ[i, j] * Δxᶜᶠᵃ[i, j]

        # Face - Face areas are calculated as the Center - Center ones
        a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1])
        b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1])
        c = lat_lon_to_cartesian(φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ])
        d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ])

        Azᶠᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2
    end
end
