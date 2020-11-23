@inline fractional_x_index(x, ::Type{Cell}, grid::RegularCartesianGrid) = @inbounds (x - grid.xC[1]) / grid.Δx
@inline fractional_x_index(x, ::Type{Face}, grid::RegularCartesianGrid) = @inbounds (x - grid.xF[1]) / grid.Δx

@inline fractional_y_index(y, ::Type{Cell}, grid::RegularCartesianGrid) = @inbounds (y - grid.yC[1]) / grid.Δy
@inline fractional_y_index(y, ::Type{Face}, grid::RegularCartesianGrid) = @inbounds (y - grid.yF[1]) / grid.Δy

@inline fractional_z_index(z, ::Type{Cell}, grid::RegularCartesianGrid) = @inbounds (z - grid.zC[1]) / grid.Δz
@inline fractional_z_index(z, ::Type{Face}, grid::RegularCartesianGrid) = @inbounds (z - grid.zF[1]) / grid.Δz

"""
    fractional_indices(x, y, z, loc, grid::RegularCartesianGrid)

Convert the coordinates `(x, y, z)` to _fractional_ indices on a regular Cartesian grid located at `loc`
where `loc` is a 3-tuple of `Cell` and `Face`. Fractional indices are floats indicating a location between
grid points.
"""
@inline function fractional_indices(x, y, z, loc, grid::RegularCartesianGrid)
    i = fractional_x_index(x, loc[1], grid)
    j = fractional_y_index(y, loc[2], grid)
    k = fractional_z_index(z, loc[3], grid)
    return i, j, k
end

# Trilinear Lagrange polynomials
@inline ϕ₁(ξ, η, ζ) = (1 - ξ) * (1 - η) * (1 - ζ)
@inline ϕ₂(ξ, η, ζ) = (1 - ξ) * (1 - η) * ζ
@inline ϕ₃(ξ, η, ζ) = (1 - ξ) * η * (1 - ζ)
@inline ϕ₄(ξ, η, ζ) = (1 - ξ) * η * ζ
@inline ϕ₅(ξ, η, ζ) = ξ * (1 - η) * (1 - ζ)
@inline ϕ₆(ξ, η, ζ) = ξ * (1 - η) * ζ
@inline ϕ₇(ξ, η, ζ) = ξ * η * (1 - ζ)
@inline ϕ₈(ξ, η, ζ) = ξ * η * ζ

@inline _interpolate(field, ξ, η, ζ, i, j, k) =
    @inbounds (  ϕ₁(ξ, η, ζ) * field[i,   j,   k  ]
               + ϕ₂(ξ, η, ζ) * field[i,   j,   k+1]
               + ϕ₃(ξ, η, ζ) * field[i,   j+1, k  ]
               + ϕ₄(ξ, η, ζ) * field[i,   j+1, k+1]
               + ϕ₅(ξ, η, ζ) * field[i+1, j,   k  ]
               + ϕ₆(ξ, η, ζ) * field[i+1, j,   k+1]
               + ϕ₇(ξ, η, ζ) * field[i+1, j+1, k  ]
               + ϕ₈(ξ, η, ζ) * field[i+1, j+1, k+1])

"""
    interpolate(field, x, y, z)

Interpolate `field` to the physical point `(x, y, z)` using trilinear interpolation.
"""
@inline function interpolate(field, x, y, z)
    i, j, k = fractional_indices(x, y, z, location(field), field.grid)

    # Convert fractional indices to unit cell coordinates 0 <= (ξ, η, ζ) <=1
    # and integer indices (with 0-based indexing).
    ξ, i = modf(i)
    η, j = modf(j)
    ζ, k = modf(k)

    # Convert indices to proper integers and shift to 1-based indexing.
    return _interpolate(field.data, ξ, η, ζ, Int(i+1), Int(j+1), Int(k+1))
end

@inline function interpolate(field, LX, LY, LZ, grid, x, y, z)
    i, j, k = fractional_indices(x, y, z, (LX, LY, LZ), grid)

    # Use mod and trunc as CUDA.modf is not defined.
    ξ, i = mod(i, 1), Base.unsafe_trunc(Int, i)
    η, j = mod(j, 1), Base.unsafe_trunc(Int, j)
    ζ, k = mod(k, 1), Base.unsafe_trunc(Int, k)

    return _interpolate(field, ξ, η, ζ, i+1, j+1, k+1)
end

