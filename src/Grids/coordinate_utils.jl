abstract type CallableCoordinate end

# Callable coordinates can be indexed just like arrays!
Base.getindex(coord::CallableCoordinate, i) = coord(i)

struct ExponentialCoordinate <: CallableCoordinate
    size :: Int
    faces :: Vector{Float64}
    left :: Float64
    right :: Float64
    scale :: Float64
    bias :: Symbol
    function ExponentialCoordinate(size::Int, left::Number, right::Number, scale::Number, bias::Symbol)
        faces = [construct_exponential_coordinate(i, size, left, right, scale, bias) for i in 1:size+1]
        return new(size, faces, left, right, scale, bias)
    end
end

# An exponential coordinate actually has faces
Base.getindex(coord::ExponentialCoordinate, i) = @inbounds coord.faces[i]

"""
    ExponentialCoordinate(N::Int, left, right;
                          scale = (right-left)/5,
                          bias = :right)

Return a one-dimensional coordinate with `N` cells that are exponentially spaced
(or, equivalently, with spacings that grow linearly along the coordinate).
The coordinate spans the range [`left`, `right`]. The exponential e-folding is controlled by `scale`.
The coordinate interfaces are closely stacked on the `bias`-side of the domain.

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

# output
ExponentialCoordinate
├─ size: 10
├─ faces: [-1000.0, -564.247649441104, -299.95048878528615, -139.64615757253702, -42.41666580727582, 16.55600197663209, 52.324733072619736, 74.0195651413529, 87.17814594835643, 95.15922864611028, 100.0]
├─ left: -1000.0
├─ right: 100.0
├─ scale: 220.0
└─ bias: :right
```

To inspect the interfaces of the coordinate we can call:

```jldoctest ExponentialCoordinate
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

Above, the default `bias` is `:right` and thus the interfaces are closer on the `right = 100` side of the domain.
We can get a left-biased grid via:

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
ExponentialCoordinate(size::Int, left, right;
                      scale = (right-left)/5,
                      bias = :right) = ExponentialCoordinate(size, left, right, scale, bias)

@inline rightbiased_exponential_mapping(x, l, r, h) = @. r - (r - l) * expm1((r - x) / h) / expm1((r - l) / h)
@inline  leftbiased_exponential_mapping(x, l, r, h) = @. l + (r - l) * expm1((x - l) / h) / expm1((r - l) / h)

function (coord::ExponentialCoordinate)(i)
    N, left, right, scale = coord.size, coord.left, coord.right, coord.scale
    return construct_exponential_coordinate(i, N, left, right, scale, coord.bias)
end

function construct_exponential_coordinate(i, N, left, right, scale, bias)

    # uniform coordinate
    Δ = (right - left) / N    # spacing
    ξᵢ = left + (i-1) * Δ     # interfaces

    # mapped coordinate
    if bias === :right
       xᵢ = rightbiased_exponential_mapping(ξᵢ, left, right, scale)
    elseif bias === :left
       xᵢ =  leftbiased_exponential_mapping(ξᵢ, left, right, scale)
    end

    if abs(xᵢ - left) < 10eps(Float32)
        xᵢ = left
    elseif abs(xᵢ - right) < 10eps(Float32)
        xᵢ = right
    end

    return xᵢ
end

Base.length(coord::ExponentialCoordinate) = coord.size

Base.summary(::ExponentialCoordinate) = "ExponentialCoordinate"

function Base.show(io::IO, coord::ExponentialCoordinate)
    return print(io, summary(coord), '\n',
                 "├─ size: ", coord.size, '\n',
                 "├─ faces: ", coord.faces, '\n',
                 "├─ left: ", coord.left, '\n',
                 "├─ right: ", coord.right, '\n',
                 "├─ scale: ", coord.scale, '\n',
                 "└─ bias: :$(coord.bias)")
end

"""
    PowerLawStretching{T}

Α power-law stretching of the form `x ↦ x^power`.
"""
struct PowerLawStretching{T}
    power :: T
end

"""
    (stretching::PowerLawStretching)(x)

Apply power-law stretching to `x` via

    x^stretching.power
"""
(stretching::PowerLawStretching)(x) = x^stretching.power

"""
    LinearStretching{T}

Α linear stretching of the form `x ↦ (1 + coefficient) * x`.
"""
struct LinearStretching{T}
    coefficient :: T
end

"""
    (stretching::LinearStretching)(x)

Apply linear stretching to `x` via

    (1 + stretching.coefficient) * x
"""
(stretching::LinearStretching)(x) = (1 + stretching.coefficient) * x

struct ConstantToStretchedCoordinate{S, A} <: CallableCoordinate
    extent :: Float64
    bias :: Symbol
    bias_edge :: Float64
    constant_spacing :: Float64
    constant_spacing_extent :: Float64
    maximum_stretching_extent :: Float64
    maximum_spacing :: Float64
    stretching :: S
    faces :: A

    function ConstantToStretchedCoordinate(extent,
                                           bias,
                                           bias_edge,
                                           constant_spacing,
                                           constant_spacing_extent,
                                           maximum_stretching_extent,
                                           maximum_spacing,
                                           stretching;
                                           rounding_digits=2)

        interfaces = compute_stretched_interfaces(; extent,
                                                  bias,
                                                  bias_edge,
                                                  constant_spacing,
                                                  constant_spacing_extent,
                                                  maximum_stretching_extent,
                                                  maximum_spacing,
                                                  stretching,
                                                  rounding_digits)
        S = typeof(stretching)
        A = typeof(interfaces)
        return new{S, A}(extent, bias, bias_edge, constant_spacing,
                         constant_spacing_extent, maximum_stretching_extent,
                         maximum_spacing, stretching, interfaces)
    end
end

function compute_stretched_interfaces(; extent,
                                      bias,
                                      bias_edge,
                                      constant_spacing,
                                      constant_spacing_extent,
                                      maximum_stretching_extent,
                                      maximum_spacing,
                                      stretching = PowerLawStretching(1.02),
                                      rounding_digits)

    Δ₀ = constant_spacing
    h₀ = constant_spacing_extent

    dir = bias === :left ? 1 :
          bias === :right ? -1 :
          throw(ArgumentError("bias must be :left or :right"))

    # Generate surface layer grid
    faces = [bias_edge + dir * Δ₀ * (i-1) for i = 1:ceil(h₀ / Δ₀)+1]

    # Generate stretched interior grid
    L₀ = extent

    while abs(faces[end] - bias_edge) < L₀
        Δ_previous = abs(faces[end] - faces[end-1])

        if abs(bias_edge - faces[end]) ≤ maximum_stretching_extent
            Δ = stretching(Δ_previous)
            Δ = min(maximum_spacing, Δ)
        else
            Δ = Δ_previous
        end

        push!(faces, round(faces[end] + dir * Δ, digits=rounding_digits))
    end

    if dir == -1
        faces = reverse(faces)
    end

    return faces
end

"""
    ConstantToStretchedCoordinate(; extent,
                                  bias = :right,
                                  bias_edge = 0,
                                  constant_spacing = extent / 20,
                                  constant_spacing_extent = 5 * constant_spacing,
                                  maximum_stretching_extent = Inf,
                                  maximum_spacing = Inf,
                                  stretching = PowerLawStretching(1.02),
                                  rounding_digits = 2)

Return a one-dimensional coordinate that has `constant_spacing` over a `constant_spacing_extent`
on the `bias`-side of the domain.
The coordinate has constant spacing over a distance

    ceil(constant_spacing_extent / constant_spacing) * constant_spacing > constant_spacing_extent

from the `bias_edge`.
Beyond the above distance, the interface spacings stretch according the provided `stretching` law.

Keyword arguments
=================

* `extent`: The desired extent of the coordinate.
* `bias :: Symbol`: Whether the `constant_spacing` interfaces are on the left (`:left`) or right (`:right`)
  part of the domain. Default: `:right`.
* `bias_edge`: The first interface on the `bias`-side of the domain. Default: 0.
* `constant_spacing`: The constant spacing on the `bias`-side of the domain. Default: `extent / 20`.
* `constant_spacing_extent`: The extent of the domain away from the `bias_edge` for which we have
  `constant_spacing`. Default: `5 * constant_spacing`.
* `maximum_stretching_extent`: The distance away from the `bias_edge` beyond which there is no more stretching
  and instead we transition to a uniformly-spaced coordinate. Default: Inf.
* `maximum_spacing`: The maximum spacing between two interfaces. Default: Inf.
* `stretching`: The stretching law. Available options are [`PowerLawStretching`](@ref) and [`LinearStretching`](@ref).
   Default: `PowerLawStretching(1.02)`.
* `rounding_digits`: the accuracy with which the grid interfaces are saved. Default: 2.

Examples
========

* A vertical coordinate with constant 20-meter spacing at the top 110 meters.
  For that, we use the defaults `bias = :right` and `bias_edge = 0`.

  ```jldoctest ConstantToStretchedCoordinate
  using Oceananigans

  z = ConstantToStretchedCoordinate(extent = 200,
                                    constant_spacing = 25,
                                    constant_spacing_extent = 90)
  # output
  ConstantToStretchedCoordinate
  ├─ extent: 200.0
  ├─ bias: :right
  ├─ bias_edge: 0.0
  ├─ constant_spacing: 25.0
  ├─ constant_spacing_extent: 90.0
  ├─ maximum_stretching_extent: Inf
  ├─ maximum_spacing: Inf
  ├─ stretching: PowerLawStretching{Float64}(1.02)
  └─ faces: : 9-element Vector{Float64}
  ```

  The `z` coordinate above has

  ```jldoctest ConstantToStretchedCoordinate
  N = length(z)

  # output
  8
  ```

  cells. The coordinate's interfaces are:

  ```jldoctest ConstantToStretchedCoordinate
  z.faces

  # output
  9-element Vector{Float64}:
   -218.16
   -185.57
   -155.13
   -126.66
   -100.0
    -75.0
    -50.0
    -25.0
      0.0
  ```

  The coordinate has an extent that is longer from what prescribed via the `extent`
  keyword argument, namely by:

  ```jldoctest ConstantToStretchedCoordinate
  (z.faces[end] - z.faces[1]) - z.extent

  # output

  18.159999999999997
  ```

* A coordinate that that has a 20-meter spacing for 50 meters at the left side of the domain.
  The left-most interface of the domain is at -50 meters and the coordinate extends for at least 250 meters.

  ```jldoctest ConstantToStretchedCoordinate
  using Oceananigans

  x = ConstantToStretchedCoordinate(extent = 250,
                                    bias = :left,
                                    bias_edge = -50,
                                    constant_spacing = 20,
                                    constant_spacing_extent = 50)

  x.faces

  # output

  12-element Vector{Float64}:
   -50.0
   -30.0
   -10.0
    10.0
    31.23
    53.8
    77.82
   103.42
   130.74
   159.93
   191.16
   224.62
  ```

  that ends up with

  ```jldoctest ConstantToStretchedCoordinate
  length(x)

  # output
  11
  ```

  cells that span a domain of:

  ```jldoctest ConstantToStretchedCoordinate
  x.faces[end] - x.faces[1]

  # output
  274.62
  ```
  which is bigger than the desired `extent`.
"""
function ConstantToStretchedCoordinate(; extent = 1000,
                                       bias = :right,
                                       bias_edge = 0,
                                       constant_spacing = extent / 20,
                                       constant_spacing_extent = 5 * constant_spacing,
                                       maximum_stretching_extent = Inf,
                                       maximum_spacing = Inf,
                                       stretching = PowerLawStretching(1.02),
                                       rounding_digits = 2)

    return ConstantToStretchedCoordinate(extent,
                                         bias,
                                         bias_edge,
                                         constant_spacing,
                                         constant_spacing_extent,
                                         maximum_stretching_extent,
                                         maximum_spacing,
                                         stretching;
                                         rounding_digits)
end

(coord::ConstantToStretchedCoordinate)(i) = coord.faces[i]

Base.length(coord::ConstantToStretchedCoordinate) = length(coord.faces)-1

Base.summary(::ConstantToStretchedCoordinate) = "ConstantToStretchedCoordinate"

function Base.show(io::IO, coord::ConstantToStretchedCoordinate)
    return print(io, summary(coord), '\n',
                 "├─ extent: ", coord.extent, '\n',
                 "├─ bias: :$(coord.bias)", '\n',
                 "├─ bias_edge: ", coord.bias_edge, '\n',
                 "├─ constant_spacing: ", coord.constant_spacing, '\n',
                 "├─ constant_spacing_extent: ", coord.constant_spacing_extent, '\n',
                 "├─ maximum_stretching_extent: ", coord.maximum_stretching_extent, '\n',
                 "├─ maximum_spacing: ", coord.maximum_spacing, '\n',
                 "├─ stretching: ", coord.stretching, '\n',
                 "└─ faces: : ", summary(coord.faces))
end
