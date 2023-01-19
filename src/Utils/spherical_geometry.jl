using LinearAlgebra: dot, cross

"""
    spherical_area_triangle(a::Number, b::Number, c::Number)

Return the area of a spherical triangle on the unit sphere with sides `a`, `b`, and `c`.

The area of a spherical triangle on the unit sphere is ``E = A + B + C - π``, where ``A``, ``B``, and ``C``
are the triangle's inner angles.

It has been known since Euler and Lagrange that ``\\tan(E/2) = P / (1 + \\cos a + \\cos b + \\cos c)``, where
``P = (1 - \\cos²a - \\cos²b - \\cos²c + 2 \\cos a \\cos b \\cos c)^{1/2}``.
"""
function spherical_area_triangle(a::Number, b::Number, c::Number)
    cosa, cosb, cosc = cos.((a, b, c))

    tan½E = sqrt(1 - cosa^2 - cosb^2 - cosc^2 + 2cosa * cosb * cosc)
    tan½E /= 1 + cosa + cosb + cosc

    return 2atan(tan½E)
end

"""
    spherical_area_triangle(a::AbstractVector, b::AbstractVector, c::AbstractVector)

Return the area of a spherical triangle on the unit sphere with vertices given by the 3-vectors
`a`, `b`, and `c` whose origin is the the center of the sphere. The formula was first given by
Eriksson (1990).

If we denote with ``A``, ``B``, and ``C`` the inner angles of the spherical triangle and with
``a``, ``b``, and ``c`` the side of the triangle then, it has been known since Euler and Lagrange
that ``\\tan(E/2) = P / (1 + \\cos a + \\cos b + \\cos c)``, where ``E = A + B + C - π`` is the
triangle's excess and ``P = (1 - \\cos²a - \\cos²b - \\cos²c + 2 \\cos a \\cos b \\cos c)^{1/2}``.
On the unit sphere, ``E`` is precisely the area of the spherical triangle. Erikkson (1990) showed
that ``P`` above  the same as the volume defined by the vectors `a`, `b`, and `c`, that is
``P = |𝐚 \\cdot (𝐛 \\times 𝐜)|``.

References
==========
Eriksson, F. (1990) On the measure of solid angles, Mathematics Magazine, 63 (3), 184-187, doi:10.1080/0025570X.1990.11977515
"""
function spherical_area_triangle(a₁::AbstractVector, a₂::AbstractVector, a₃::AbstractVector)
    (sum(a₁.^2) ≈ 1 && sum(a₂.^2) ≈ 1 && sum(a₃.^2) ≈ 1) || error("a₁, a₂, a₃ must be unit vectors")

    tan½E = abs(dot(a₁, cross(a₂, a₃)))
    tan½E /= 1 + dot(a₁, a₂) + dot(a₂, a₃) + dot(a₁, a₃)

    return 2atan(tan½E)
end

"""
    hav(x)

Compute haversine of `x`, where `x` is in radians: `hav(x) = sin²(x/2)`.
"""
hav(x) = sin(x/2)^2

"""
    central_angle((φ₁, λ₁), (φ₂, λ₂))

Compute the central angle (in radians) between two points on the sphere with
`(latitude, longitude)` coordinates `(φ₁, λ₁)` and `(φ₂, λ₂)` (in radians).

References
==========
- [Wikipedia, Great-circle distance](https://en.wikipedia.org/wiki/Great-circle_distance)
"""
function central_angle((φ₁, λ₁), (φ₂, λ₂))
    Δφ, Δλ = φ₁ - φ₂, λ₁ - λ₂

    return 2asin(sqrt(hav(Δφ) + (1 - hav(Δφ) - hav(φ₁ + φ₂)) * hav(Δλ)))
end

"""
    central_angle_degrees((φ₁, λ₁), (φ₂, λ₂))

Compute the central angle (in degrees) between two points on the sphere with
`(latitude, longitude)` coordinates `(φ₁, λ₁)` and `(φ₂, λ₂)` (in degrees).

See also [`central_angle`](@ref).
"""
central_angle_degrees((φ₁, λ₁), (φ₂, λ₂)) = rad2deg(central_angle(deg2rad.((φ₁, λ₁)), deg2rad.((φ₂, λ₂))))
