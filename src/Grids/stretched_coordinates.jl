struct ExponentialCoordinate <: Function
    size :: Int
    left :: Float64
    right :: Float64
    scale :: Float64
    bias :: Symbol
end

"""
    ExponentialCoordinate(N::Int, left, right; scale=(right-left)/5, bias=:right)

Return a type that describes a one-dimensional coordinate with `N` cells that
are exponentially spaced (or, equivalently, with spacings that grow linearly).

The coordinate spans the range [`left`, `right`]. The exponential e-folding is controlled by `scale`.
The coordinate interfaces are stacked on the `bias`-side of the domain.

Arguments
=========
- `N`: The number of cells in the coordinate.
- `left`: The left-most interface of the coordinate.
- `right`: The right-most interface of the coordinate.

Keyword Arguments
=================
- `scale`: The length scale of the exponential e-folding. Default: `(right - left) / 5`
- `bias :: Symbol`: Determine whether left or right biased. Default: `:right`.

Examples
========

```jldoctest ExponentialCoordinate
using Oceananigans

N = 10
l = -1000
r = 100

x = ExponentialCoordinate(N, l, r)

[x(i) for i in 1:N+1]

# output

11-element Vector{Float64}:
 -1000.0
  -564.247649441104
  -299.95048878528615
  -139.64615757253702
   -42.41666580727582
    16.55600197663209
    52.324733072619736
    74.0195651413529
    87.17814594835643
    95.15922864611028
   100.0
```

Above, the default `bias` is `:right`. We can get a left-biased grid via:

```jldoctest ExponentialCoordinate
x = ExponentialCoordinate(N, l, r, bias=:left)

[x(i) for i in 1:N+1]

# output

11-element Vector{Float64}:
 -1000.0
  -995.1592286461103
  -987.1781459483565
  -974.0195651413529
  -952.3247330726198
  -916.556001976632
  -857.5833341927241
  -760.353842427463
  -600.0495112147139
  -335.75235055889596
   100.0
```
"""
ExponentialCoordinate(size::Int, left, right; scale=(right-left)/5, bias=:right) =
    ExponentialCoordinate(size, left, right, scale, bias)

@inline rightbiased_exponential_mapping(x, l, r, h) = @. r - (r - l) * expm1((r - x) / h) / expm1((r - l) / h)
@inline  leftbiased_exponential_mapping(x, l, r, h) = @. l + (r - l) * expm1((x - l) / h) / expm1((r - l) / h)

function (coord::ExponentialCoordinate)(i)
    N, left, right, scale = coord.size, coord.left, coord.right, coord.scale

    # uniform coordinate
    ξᵢ = left + (i-1) * (right - left) / N

    # mapped coordinate
    if coord.bias === :right
       xᵢ = rightbiased_exponential_mapping(ξᵢ, left, right, scale)
    elseif coord.bias === :left
       xᵢ = leftbiased_exponential_mapping(ξᵢ, left, right, scale)
    end

    if abs(xᵢ - left) < 10eps(Float32)
        xᵢ = left
    elseif abs(xᵢ - right) < 10eps(Float32)
        xᵢ = right
    end

    return xᵢ
end
