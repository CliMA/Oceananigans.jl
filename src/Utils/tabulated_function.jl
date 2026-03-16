using Adapt
using KernelAbstractions: @kernel, @index
import KernelAbstractions
using DocStringExtensions: TYPEDFIELDS, TYPEDSIGNATURES
using Oceananigans.Architectures: CPU, device
import Oceananigans.Architectures: on_architecture

"""
    $(TYPEDFIELDS)

A wrapper around a callable that precomputes values in an N-dimensional lookup table
for fast interpolation. Supports 1D (linear), 2D (bilinear), and 3D (trilinear) interpolation.
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
    $(TYPEDSIGNATURES)

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

    normalized_range = _normalize_range(range)
    N = length(normalized_range)
    normalized_points = _normalize_points(points, Val(N))

    # Compute grid spacings
    inverse_Δ = map(normalized_range, normalized_points) do r, p
        Δ = (r[2] - r[1]) / (p - 1)
        convert(FT, 1 / Δ)
    end

    # Convert range tuples to FT
    converted_range = map(r -> (convert(FT, r[1]), convert(FT, r[2])), normalized_range)

    # Build lookup table directly on the target architecture
    table = build_table(arch, func, converted_range, normalized_points, inverse_Δ)

    return TabulatedFunction{N, typeof(func), typeof(table), typeof(converted_range), typeof(inverse_Δ)}(
        func, table, converted_range, inverse_Δ)
end

#####
##### Table building kernels
#####

@kernel function _build_table_1d_kernel!(table, func, range, inverse_Δ)
    i = @index(Global)
    @inbounds begin
        x_min = range[1][1]
        inv_Δx = inverse_Δ[1]
        table[i] = func(x_min + (i - 1) / inv_Δx)
    end
end

@kernel function _build_table_2d_kernel!(table, func, range, inverse_Δ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        x_rng, y_rng = range
        x_min, y_min = x_rng[1], y_rng[1]
        inv_Δx, inv_Δy = inverse_Δ
        table[i, j] = func(x_min + (i - 1) / inv_Δx,
                           y_min + (j - 1) / inv_Δy)
    end
end

@kernel function _build_table_3d_kernel!(table, func, range, inverse_Δ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        x_rng, y_rng, z_rng = range
        x_min, y_min, z_min = x_rng[1], y_rng[1], z_rng[1]
        inv_Δx, inv_Δy, inv_Δz = inverse_Δ
        table[i, j, k] = func(x_min + (i - 1) / inv_Δx,
                              y_min + (j - 1) / inv_Δy,
                              z_min + (k - 1) / inv_Δz)
    end
end

#####
##### Table builders for each dimensionality
#####

function build_table(arch, func, range::NTuple{1}, points::NTuple{1}, inverse_Δ)
    dev = device(arch)
    FT = eltype(inverse_Δ)
    table = KernelAbstractions.zeros(dev, FT, points...)
    kernel! = _build_table_1d_kernel!(dev, 256)
    kernel!(table, func, range, inverse_Δ; ndrange=points)
    return table
end

function build_table(arch, func, range::NTuple{2}, points::NTuple{2}, inverse_Δ)
    dev = device(arch)
    FT = eltype(inverse_Δ)
    table = KernelAbstractions.zeros(dev, FT, points...)
    kernel! = _build_table_2d_kernel!(dev, (16, 16))
    kernel!(table, func, range, inverse_Δ; ndrange=points)
    return table
end

function build_table(arch, func, range::NTuple{3}, points::NTuple{3}, inverse_Δ)
    dev = device(arch)
    FT = eltype(inverse_Δ)
    table = KernelAbstractions.zeros(dev, FT, points...)
    kernel! = _build_table_3d_kernel!(dev, (8, 8, 8))
    kernel!(table, func, range, inverse_Δ; ndrange=points)
    return table
end

# Interpolation utilities (interpolator, _interpolate, ϕ₁-ϕ₈) are defined in interpolation.jl

#####
##### Evaluation: 1D linear interpolation
#####

@inline function (f::TabulatedFunction1D)(x)
    x_min, x_max = f.range[1]
    x_clamped = clamp(x, x_min, x_max)

    fractional_idx = (x_clamped - x_min) * f.inverse_Δ[1]
    i⁻, i⁺, ξ = interpolator(fractional_idx)

    n = length(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, n)

    return _interpolate(f.table, (i⁻, i⁺, ξ))
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

    i⁻, i⁺, ξ = interpolator(frac_i)
    j⁻, j⁺, η = interpolator(frac_j)

    nx, ny = size(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, nx)
    j⁻ = j⁻ + 1
    j⁺ = min(j⁺ + 1, ny)

    return _interpolate(f.table, (i⁻, i⁺, ξ), (j⁻, j⁺, η))
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

    i⁻, i⁺, ξ = interpolator(frac_i)
    j⁻, j⁺, η = interpolator(frac_j)
    k⁻, k⁺, ζ = interpolator(frac_k)

    nx, ny, nz = size(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, nx)
    j⁻ = j⁻ + 1
    j⁺ = min(j⁺ + 1, ny)
    k⁻ = k⁻ + 1
    k⁺ = min(k⁺ + 1, nz)

    return _interpolate(f.table, (i⁻, i⁺, ξ), (j⁻, j⁺, η), (k⁻, k⁺, ζ))
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
