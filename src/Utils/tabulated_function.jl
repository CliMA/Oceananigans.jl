using Adapt
using Oceananigans.Architectures: CPU
import Oceananigans.Architectures: on_architecture

"""
    TabulatedFunction{N, F, T, R, D}

A wrapper around a callable that precomputes values in an N-dimensional lookup table
for fast interpolation. Supports 1D (linear), 2D (bilinear), and 3D (trilinear) interpolation.

# Type Parameters
- `N`: Dimensionality (1, 2, or 3)
- `F`: Type of the original function
- `T`: Type of the lookup table
- `R`: Type of the range specification
- `D`: Type of the inverse grid spacing(s)

# Fields
- `func`: The original callable being tabulated
- `table`: Precomputed values (N-dimensional array)
- `range`: Tuple of (min, max) for each dimension
- `inverse_Δ`: Inverse grid spacing for each dimension

# Example

```jldoctest tabulatedfunc
using Oceananigans.Utils: TabulatedFunction

# Tabulate a simple function for fast evaluation
f = TabulatedFunction(sin; range=(0, 2π))

# output
TabulatedFunction{1} with 100 points over [0.0, 6.283185307179586] of sin
```

```jldoctest tabulatedfunc
# Evaluate like a regular function
f(π/2)

# output
0.9996224305511583
```

2D tabulation:

```jldoctest
using Oceananigans.Utils: TabulatedFunction

g(x, y) = sin(x) * cos(y)
f = TabulatedFunction(g; range=((0, π), (0, 2π)), points=(50, 100))

# output
TabulatedFunction{2} with 50×100 points over [0.0, 3.141592653589793] × [0.0, 6.283185307179586] of g
```

3D tabulation:

```jldoctest
using Oceananigans.Utils: TabulatedFunction

h(x, y, z) = x^2 + y^2 + z^2
f = TabulatedFunction(h; range=((-1, 1), (-1, 1), (-1, 1)), points=20)

# output
TabulatedFunction{3} with 20×20×20 points over [-1.0, 1.0] × [-1.0, 1.0] × [-1.0, 1.0] of h
```
"""
struct TabulatedFunction{N, F, T, R, D}
    func :: F
    table :: T
    range :: R
    inverse_Δ :: D
end

# Type aliases for dispatch
const TabulatedFunction1D = TabulatedFunction{1}
const TabulatedFunction2D = TabulatedFunction{2}
const TabulatedFunction3D = TabulatedFunction{3}

#####
##### Dimensionality detection and normalization
#####

# Detect dimensionality from range specification
# 1D: range = (a, b) where a and b are numbers
_tabulated_ndims(::Tuple{<:Number, <:Number}) = 1
# 2D: range = ((x1, x2), (y1, y2)) - tuple of two 2-tuples
_tabulated_ndims(::Tuple{<:Tuple, <:Tuple}) = 2
# 3D: range = ((x1, x2), (y1, y2), (z1, z2)) - tuple of three 2-tuples
_tabulated_ndims(::Tuple{<:Tuple, <:Tuple, <:Tuple}) = 3

# Normalize range to tuple-of-tuples format (internal representation)
_normalize_range(range::Tuple{<:Number, <:Number}) = (range,)
_normalize_range(range::Tuple{<:Tuple, <:Tuple}) = range
_normalize_range(range::Tuple{<:Tuple, <:Tuple, <:Tuple}) = range

# Normalize points to tuple format
_normalize_points(points::Integer, ::Val{N}) where N = ntuple(_ -> points, Val(N))
_normalize_points(points::NTuple{N, <:Integer}, ::Val{N}) where N = points

#####
##### Constructor
#####

"""
    TabulatedFunction(func, [arch=CPU()], [FT=Float64]; range, points=100)

Construct a `TabulatedFunction` by precomputing values over the specified range(s)
for fast linear, bilinear, or trilinear interpolation.

# Arguments
- `func`: Callable taking 1, 2, or 3 numeric arguments
- `arch`: Architecture for the lookup table (`CPU()` or `GPU()`)
- `FT`: Float type for table values

# Keyword Arguments
- `range`: For 1D: `(min, max)`. For 2D: `((x_min, x_max), (y_min, y_max))`.
           For 3D: `((x_min, x_max), (y_min, y_max), (z_min, z_max))`.
- `points`: Number of points per dimension. Scalar (applied to all dims) or tuple.

# Examples

```jldoctest
using Oceananigans.Utils: TabulatedFunction

# 1D: Tabulate trigonometric function
f = TabulatedFunction(sin; range=(0, 2π), points=1000)
f(π/4)

# output
0.7071052539107768
```

```jldoctest
using Oceananigans.Utils: TabulatedFunction

# 2D: Tabulate a function of two variables
g(x, y) = x^2 + y^2
f = TabulatedFunction(g; range=((-1, 1), (-1, 1)), points=50)
f(0.5, 0.5)

# output
0.5006247396917951
```

```jldoctest
using Oceananigans.Utils: TabulatedFunction

# 3D: Tabulate a function of three variables
h(x, y, z) = x * y * z
f = TabulatedFunction(h; range=((0, 1), (0, 1), (0, 1)), points=(10, 10, 10))
f(0.5, 0.5, 0.5)

# output
0.125
```

Values outside `range` are clamped to the nearest table boundary.
"""
function TabulatedFunction(func, arch=CPU(), FT=Oceananigans.defaults.FloatType;
                           range,
                           points = 100)

    N = _tabulated_ndims(range)
    normalized_range = _normalize_range(range)
    normalized_points = _normalize_points(points, Val(N))

    # Compute grid spacings
    inverse_Δ = ntuple(Val(N)) do d
        r = normalized_range[d]
        p = normalized_points[d]
        Δ = (r[2] - r[1]) / (p - 1)
        convert(FT, 1 / Δ)
    end

    # Build lookup table
    table = _build_table(func, FT, Val(N), normalized_range, normalized_points, inverse_Δ)
    table = on_architecture(arch, table)

    # Convert range tuples to FT
    converted_range = map(r -> (convert(FT, r[1]), convert(FT, r[2])), normalized_range)

    return TabulatedFunction{N, typeof(func), typeof(table), typeof(converted_range), typeof(inverse_Δ)}(
        func, table, converted_range, inverse_Δ)
end

#####
##### Table builders for each dimensionality
#####

@inline function _build_table(func, FT, ::Val{1}, range, points, inverse_Δ)
    x_min = range[1][1]
    inv_Δx = inverse_Δ[1]
    return [convert(FT, func(x_min + (i - 1) / inv_Δx)) for i in 1:points[1]]
end

@inline function _build_table(func, FT, ::Val{2}, range, points, inverse_Δ)
    x_min, y_min = range[1][1], range[2][1]
    inv_Δx, inv_Δy = inverse_Δ
    return [convert(FT, func(x_min + (i - 1) / inv_Δx,
                             y_min + (j - 1) / inv_Δy))
            for i in 1:points[1], j in 1:points[2]]
end

@inline function _build_table(func, FT, ::Val{3}, range, points, inverse_Δ)
    x_min, y_min, z_min = range[1][1], range[2][1], range[3][1]
    inv_Δx, inv_Δy, inv_Δz = inverse_Δ
    return [convert(FT, func(x_min + (i - 1) / inv_Δx,
                             y_min + (j - 1) / inv_Δy,
                             z_min + (k - 1) / inv_Δz))
            for i in 1:points[1], j in 1:points[2], k in 1:points[3]]
end

#####
##### Interpolation helper
#####

@inline function _tabulated_interpolator(fractional_idx)
    # For why we use Base.unsafe_trunc instead of trunc see:
    # https://github.com/CliMA/Oceananigans.jl/issues/828
    # https://github.com/CliMA/Oceananigans.jl/pull/997
    i⁻ = Base.unsafe_trunc(Int, fractional_idx)
    i⁺ = i⁻ + 1
    ξ = mod(fractional_idx, 1)
    return (i⁻, i⁺, ξ)
end

#####
##### Evaluation: 1D linear interpolation
#####

@inline function (f::TabulatedFunction1D)(x)
    x_min, x_max = f.range[1]
    x_clamped = clamp(x, x_min, x_max)

    fractional_idx = (x_clamped - x_min) * f.inverse_Δ[1]
    i⁻, i⁺, ξ = _tabulated_interpolator(fractional_idx)

    n = length(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, n)

    f⁻ = @inbounds f.table[i⁻]
    f⁺ = @inbounds f.table[i⁺]

    return (1 - ξ) * f⁻ + ξ * f⁺
end

#####
##### Evaluation: 2D bilinear interpolation
#####

@inline function (f::TabulatedFunction2D)(x, y)
    x_min, x_max = f.range[1]
    y_min, y_max = f.range[2]

    x_clamped = clamp(x, x_min, x_max)
    y_clamped = clamp(y, y_min, y_max)

    frac_i = (x_clamped - x_min) * f.inverse_Δ[1]
    frac_j = (y_clamped - y_min) * f.inverse_Δ[2]

    i⁻, i⁺, ξ = _tabulated_interpolator(frac_i)
    j⁻, j⁺, η = _tabulated_interpolator(frac_j)

    nx, ny = size(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, nx)
    j⁻ = j⁻ + 1
    j⁺ = min(j⁺ + 1, ny)

    f₀₀ = @inbounds f.table[i⁻, j⁻]
    f₁₀ = @inbounds f.table[i⁺, j⁻]
    f₀₁ = @inbounds f.table[i⁻, j⁺]
    f₁₁ = @inbounds f.table[i⁺, j⁺]

    # Bilinear interpolation
    return (1 - ξ) * (1 - η) * f₀₀ +
                 ξ * (1 - η) * f₁₀ +
           (1 - ξ) *       η * f₀₁ +
                 ξ *       η * f₁₁
end

#####
##### Evaluation: 3D trilinear interpolation
#####

@inline function (f::TabulatedFunction3D)(x, y, z)
    x_min, x_max = f.range[1]
    y_min, y_max = f.range[2]
    z_min, z_max = f.range[3]

    x_clamped = clamp(x, x_min, x_max)
    y_clamped = clamp(y, y_min, y_max)
    z_clamped = clamp(z, z_min, z_max)

    frac_i = (x_clamped - x_min) * f.inverse_Δ[1]
    frac_j = (y_clamped - y_min) * f.inverse_Δ[2]
    frac_k = (z_clamped - z_min) * f.inverse_Δ[3]

    i⁻, i⁺, ξ = _tabulated_interpolator(frac_i)
    j⁻, j⁺, η = _tabulated_interpolator(frac_j)
    k⁻, k⁺, ζ = _tabulated_interpolator(frac_k)

    nx, ny, nz = size(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, nx)
    j⁻ = j⁻ + 1
    j⁺ = min(j⁺ + 1, ny)
    k⁻ = k⁻ + 1
    k⁺ = min(k⁺ + 1, nz)

    f₀₀₀ = @inbounds f.table[i⁻, j⁻, k⁻]
    f₁₀₀ = @inbounds f.table[i⁺, j⁻, k⁻]
    f₀₁₀ = @inbounds f.table[i⁻, j⁺, k⁻]
    f₁₁₀ = @inbounds f.table[i⁺, j⁺, k⁻]
    f₀₀₁ = @inbounds f.table[i⁻, j⁻, k⁺]
    f₁₀₁ = @inbounds f.table[i⁺, j⁻, k⁺]
    f₀₁₁ = @inbounds f.table[i⁻, j⁺, k⁺]
    f₁₁₁ = @inbounds f.table[i⁺, j⁺, k⁺]

    # Trilinear interpolation
    return (1 - ξ) * (1 - η) * (1 - ζ) * f₀₀₀ +
                 ξ * (1 - η) * (1 - ζ) * f₁₀₀ +
           (1 - ξ) *       η * (1 - ζ) * f₀₁₀ +
                 ξ *       η * (1 - ζ) * f₁₁₀ +
           (1 - ξ) * (1 - η) *       ζ * f₀₀₁ +
                 ξ * (1 - η) *       ζ * f₁₀₁ +
           (1 - ξ) *       η *       ζ * f₀₁₁ +
                 ξ *       η *       ζ * f₁₁₁
end

#####
##### GPU/architecture support
#####

function on_architecture(arch, f::TabulatedFunction{N}) where N
    new_table = on_architecture(arch, f.table)
    return TabulatedFunction{N, typeof(f.func), typeof(new_table), typeof(f.range), typeof(f.inverse_Δ)}(
        f.func, new_table, f.range, f.inverse_Δ)
end

# Adapt for GPU kernels (drops the original function to avoid GPU compilation issues)
function Adapt.adapt_structure(to, f::TabulatedFunction{N}) where N
    adapted_table = Adapt.adapt(to, f.table)
    return TabulatedFunction{N, Nothing, typeof(adapted_table), typeof(f.range), typeof(f.inverse_Δ)}(
        nothing, adapted_table, f.range, f.inverse_Δ)
end

#####
##### Pretty printing
#####

function Base.summary(f::TabulatedFunction{N}) where N
    dims = N == 1 ? "$(length(f.table))" : join(size(f.table), "×")
    ranges = join(["[$(r[1]), $(r[2])]" for r in f.range], " × ")
    return "TabulatedFunction{$N} with $dims points over $ranges"
end

function Base.show(io::IO, f::TabulatedFunction)
    print(io, summary(f))
    if f.func !== nothing
        print(io, " of ", f.func)
    end
end
