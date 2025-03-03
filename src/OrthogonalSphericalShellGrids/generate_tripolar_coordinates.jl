"""
    _compute_tripolar_coordinates!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC,
                                   λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                   first_pole_longitude,
                                   focal_distance, Nλ)

Compute the tripolar coordinates for a given set of input parameters. Here, we follow,
the formulation described by

> Ross J. Murray, (1996). Explicit generation of orthogonal grids for ocean models, _Journal of Computational Physics_, **126(2)**, 251-273.

The tripolar grid is built as a set of cofocal ellipsed and perpendicular hyperbolae.
The `focal_distance` argument is the distance from the center of the ellipses to the foci.

The family of ellipses obeys:

```math
\\frac{x²}{a² \\cosh² ψ} + \\frac{y²}{a² \\sinh² ψ} = 1
```

while the family of perpendicular hyperbolae obey:

```math
\\frac{x²}{a² \\mathrm{sind}² λ} - \\frac{y²}{a² \\mathrm{cosd}² λ} = 1
```

Above, ``a`` is the `focal_distance` to the center, ``λ`` is the longitudinal angle,
and ``ψ`` is the "isometric latitude", defined by Murray (1996) and satisfying:

```math
    a \\sinh ψ = \\mathrm{tand}[(90 - φ) / 2]
```

The final ``(x, y)`` points that define the stereographic projection of the tripolar
coordinates are given by:

```math
    \\begin{align*}
    x & = a \\sinh ψ \\, \\mathrm{cosd} λ \\\\
    y & = a \\sinh ψ \\, \\mathrm{sind} λ
    \\end{align*}
```

from which we can recover the longitude and latitude as:

```math
    \\begin{align*}
    λ & =    - \\frac{180}{π} \\mathrm{atan}(y / x)  \\\\
    φ & = 90 - \\frac{360}{π} \\mathrm{atan} \\sqrt{x² + y²}
    \\end{align*}
```
"""
@kernel function _compute_tripolar_coordinates!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC,
                                                λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                first_pole_longitude,
                                                focal_distance, Nλ)

    i, j = @index(Global, NTuple)

    λ2Ds = (λFF,  λFC,  λCF,  λCC)
    φ2Ds = (φFF,  φFC,  φCF,  φCC)
    λ1Ds = (λᶠᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, λᶜᵃᵃ)
    φ1Ds = (φᵃᶠᵃ, φᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ)

    for (λ2D, φ2D, λ1D, φ1D) in zip(λ2Ds, φ2Ds, λ1Ds, φ1Ds)
        ψ = asinh(tand((90 - φ1D[j]) / 2) / focal_distance)
        x = focal_distance * sind(λ1D[i]) * cosh(ψ)
        y = focal_distance * cosd(λ1D[i]) * sinh(ψ)

        # When x == 0 and y == 0 we are exactly at the north pole,
        # λ (which depends on `atan(y / x)`) is not defined
        # This makes sense, what is the longitude of the north pole? Could be anything!
        # so we choose a value that is continuous with the surrounding points.
        on_the_north_pole = (x == 0) & (y == 0)
        north_pole_value  = ifelse(i == 1, -90, 90)

        λ2D[i, j] = ifelse(on_the_north_pole, north_pole_value, - 180 / π * atan(y / x))
        φ2D[i, j] = 90 - 360 / π * atan(sqrt(y^2 + x^2)) # The latitude is in the range [-90, 90]

        # Shift longitude to the range [-180, 180],
        # the north singularities are located at -90 and 90.
        λ2D[i, j] += ifelse(i ≤ Nλ÷2, -90, 90)

        # Make sure the singularities are at longitudes we want them to be at.
        # (`first_pole_longitude` and `first_pole_longitude` + 180)
        λ2D[i, j] += first_pole_longitude + 90
        λ2D[i, j]  = convert_to_0_360(λ2D[i, j])
    end
end
