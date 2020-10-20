struct Interpolator{F}
    field :: F
end

# Converting (x, y, z) coordinates to indices on a RegularCartesianGrid.
# The returned indices can be floats indicating a location between grid points.
@inline x_to_index(x, ::Type{Cell}, grid) = @inbounds (x - grid.xC[1]) / grid.Δx
@inline x_to_index(x, ::Type{Face}, grid) = @inbounds (x - grid.xF[1]) / grid.Δx

@inline y_to_index(y, ::Type{Cell}, grid) = @inbounds (y - grid.yC[1]) / grid.Δy
@inline y_to_index(y, ::Type{Face}, grid) = @inbounds (y - grid.yF[1]) / grid.Δy

@inline z_to_index(z, ::Type{Cell}, grid) = @inbounds (z - grid.zC[1]) / grid.Δz
@inline z_to_index(z, ::Type{Face}, grid) = @inbounds (z - grid.zF[1]) / grid.Δz

@inline function coordinates_to_indices(x, y, z, loc, grid)
    i = x_to_index(x, loc[1], grid)
    j = y_to_index(y, loc[2], grid)
    k = z_to_index(z, loc[3], grid)
    return i, j, k
end

# Trilinear Lagrangian polynomials
@inline ϕ₁(ξ, η, ζ) = (1 - ξ) * (1 - η) * (1 - ζ)
@inline ϕ₂(ξ, η, ζ) = (1 - ξ) * (1 - η) * ζ
@inline ϕ₃(ξ, η, ζ) = (1 - ξ) * η * (1 - ζ)
@inline ϕ₄(ξ, η, ζ) = (1 - ξ) * η * ζ
@inline ϕ₅(ξ, η, ζ) = ξ * (1 - η) * (1 - ζ)
@inline ϕ₆(ξ, η, ζ) = ξ * (1 - η) * ζ
@inline ϕ₇(ξ, η, ζ) = ξ * η * (1 - ζ)
@inline ϕ₈(ξ, η, ζ) = ξ * η * ζ

@inline function interpolate(ξ, η, ζ, i, j, k, field)
    return (  ϕ₁(ξ, η, ζ) * field[i,   j,   k  ]
            + ϕ₂(ξ, η, ζ) * field[i,   j,   k+1]
            + ϕ₃(ξ, η, ζ) * field[i,   j+1, k+1]
            + ϕ₄(ξ, η, ζ) * field[i,   j+1, k+1]
            + ϕ₅(ξ, η, ζ) * field[i+1, j,   k  ]
            + ϕ₆(ξ, η, ζ) * field[i+1, j,   k+1]
            + ϕ₇(ξ, η, ζ) * field[i+1, j+1, k  ]
            + ϕ₈(ξ, η, ζ) * field[i+1, j+1, k+1])
end

@inline function (ℑ::Interpolator)(x, y, z, grid=ℑ.field.grid)
    i, j, k = coordinates_to_indices(x, y, z, location(ℑ.field), grid)

    # Convert fractional indices to integer indices and unit
    # cell coordinates 0 <= (ξ, η, ζ) <=1 .
    ξ, i = modf(i)
    η, j = modf(j)
    ζ, k = modf(k)

    i, j, k = Int(i), Int(j), Int(k)

    return interpolate(ξ, η, ζ, i, j, k, ℑ.field.data)
end
