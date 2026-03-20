using Adapt
using KernelAbstractions: @kernel, @index
import KernelAbstractions
using DocStringExtensions: TYPEDFIELDS, TYPEDSIGNATURES
using Oceananigans.Architectures: CPU, device
import Oceananigans.Architectures: on_architecture

"""
    $(TYPEDFIELDS)

A wrapper around a callable that precomputes values in an N-dimensional lookup table
for fast interpolation. Supports 1D (linear), 2D (bilinear), 3D (trilinear),
4D (quadrilinear), and 5D (quintilinear) interpolation.
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
const TabulatedFunction4D = TabulatedFunction{4}
const TabulatedFunction5D = TabulatedFunction{5}

#####
##### Dimensionality detection and normalization
#####

# Normalize range to tuple-of-tuples format (internal representation)
_normalize_range(range::Tuple{<:Number, <:Number}) = (range,)
_normalize_range(range::Tuple{<:Tuple, <:Tuple}) = range
_normalize_range(range::Tuple{<:Tuple, <:Tuple, <:Tuple}) = range
_normalize_range(range::Tuple{<:Tuple, <:Tuple, <:Tuple, <:Tuple}) = range
_normalize_range(range::Tuple{<:Tuple, <:Tuple, <:Tuple, <:Tuple, <:Tuple}) = range

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
- `func`: Callable taking 1, 2, 3, 4, or 5 numeric arguments
- `arch`: Architecture for the lookup table (`CPU()` or `GPU()`)
- `FT`: Float type for table values

# Keyword Arguments
- `range`: For 1D: `(min, max)`. For 2D: `((x_min, x_max), (y_min, y_max))`.
           For 3D: `((x_min, x_max), (y_min, y_max), (z_min, z_max))`.
           For 4D: `((x_min, x_max), (y_min, y_max), (z_min, z_max), (w_min, w_max))`.
           For 5D: `((x₁_min, x₁_max), ..., (x₅_min, x₅_max))`.
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

```jldoctest
using Oceananigans.Utils: TabulatedFunction

# 4D: Tabulate a function of four variables
q(x, y, z, w) = x * y * z * w
f = TabulatedFunction(q; range=((0, 1), (0, 1), (0, 1), (0, 1)), points=(10, 10, 10, 10))
f(0.5, 0.5, 0.5, 0.5)

# output
0.0625
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

@kernel function _build_table_4d_kernel!(table, func, range, inverse_Δ)
    i, j, k, l = @index(Global, NTuple)
    @inbounds begin
        x_rng, y_rng, z_rng, w_rng = range
        x_min, y_min, z_min, w_min = x_rng[1], y_rng[1], z_rng[1], w_rng[1]
        inv_Δx, inv_Δy, inv_Δz, inv_Δw = inverse_Δ
        table[i, j, k, l] = func(x_min + (i - 1) / inv_Δx,
                                  y_min + (j - 1) / inv_Δy,
                                  z_min + (k - 1) / inv_Δz,
                                  w_min + (l - 1) / inv_Δw)
    end
end

@kernel function _build_table_5d_kernel!(table, func, range, inverse_Δ)
    i, j, k, l, m = @index(Global, NTuple)
    @inbounds begin
        r₁, r₂, r₃, r₄, r₅ = range
        x₁, x₂, x₃, x₄, x₅ = r₁[1], r₂[1], r₃[1], r₄[1], r₅[1]
        d₁, d₂, d₃, d₄, d₅ = inverse_Δ
        table[i, j, k, l, m] = func(x₁ + (i - 1) / d₁,
                                     x₂ + (j - 1) / d₂,
                                     x₃ + (k - 1) / d₃,
                                     x₄ + (l - 1) / d₄,
                                     x₅ + (m - 1) / d₅)
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

function build_table(arch, func, range::NTuple{4}, points::NTuple{4}, inverse_Δ)
    dev = device(arch)
    FT = eltype(inverse_Δ)
    table = KernelAbstractions.zeros(dev, FT, points...)
    kernel! = _build_table_4d_kernel!(dev, (4, 4, 4, 4))
    kernel!(table, func, range, inverse_Δ; ndrange=points)
    return table
end

function build_table(arch, func, range::NTuple{5}, points::NTuple{5}, inverse_Δ)
    dev = device(arch)
    FT = eltype(inverse_Δ)
    table = KernelAbstractions.zeros(dev, FT, points...)
    kernel! = _build_table_5d_kernel!(dev, (4, 4, 4, 4, 4))
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
##### Evaluation: 4D quadrilinear interpolation
#####

@inline function (f::TabulatedFunction4D)(x, y, z, w)
    x_min, x_max = f.range[1]
    y_min, y_max = f.range[2]
    z_min, z_max = f.range[3]
    w_min, w_max = f.range[4]

    x_clamped = clamp(x, x_min, x_max)
    y_clamped = clamp(y, y_min, y_max)
    z_clamped = clamp(z, z_min, z_max)
    w_clamped = clamp(w, w_min, w_max)

    frac_i = (x_clamped - x_min) * f.inverse_Δ[1]
    frac_j = (y_clamped - y_min) * f.inverse_Δ[2]
    frac_k = (z_clamped - z_min) * f.inverse_Δ[3]
    frac_l = (w_clamped - w_min) * f.inverse_Δ[4]

    i⁻, i⁺, ξ = interpolator(frac_i)
    j⁻, j⁺, η = interpolator(frac_j)
    k⁻, k⁺, ζ = interpolator(frac_k)
    l⁻, l⁺, θ = interpolator(frac_l)

    nx, ny, nz, nw = size(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, nx)
    j⁻ = j⁻ + 1
    j⁺ = min(j⁺ + 1, ny)
    k⁻ = k⁻ + 1
    k⁺ = min(k⁺ + 1, nz)
    l⁻ = l⁻ + 1
    l⁺ = min(l⁺ + 1, nw)

    return _interpolate(f.table, (i⁻, i⁺, ξ), (j⁻, j⁺, η), (k⁻, k⁺, ζ), (l⁻, l⁺, θ))
end

#####
##### Evaluation: 5D quintilinear interpolation
#####

@inline function (f::TabulatedFunction5D)(x₁, x₂, x₃, x₄, x₅)
    a₁, b₁ = f.range[1]
    a₂, b₂ = f.range[2]
    a₃, b₃ = f.range[3]
    a₄, b₄ = f.range[4]
    a₅, b₅ = f.range[5]

    c₁ = clamp(x₁, a₁, b₁)
    c₂ = clamp(x₂, a₂, b₂)
    c₃ = clamp(x₃, a₃, b₃)
    c₄ = clamp(x₄, a₄, b₄)
    c₅ = clamp(x₅, a₅, b₅)

    frac_i = (c₁ - a₁) * f.inverse_Δ[1]
    frac_j = (c₂ - a₂) * f.inverse_Δ[2]
    frac_k = (c₃ - a₃) * f.inverse_Δ[3]
    frac_l = (c₄ - a₄) * f.inverse_Δ[4]
    frac_m = (c₅ - a₅) * f.inverse_Δ[5]

    i⁻, i⁺, ξ = interpolator(frac_i)
    j⁻, j⁺, η = interpolator(frac_j)
    k⁻, k⁺, ζ = interpolator(frac_k)
    l⁻, l⁺, θ = interpolator(frac_l)
    m⁻, m⁺, ψ = interpolator(frac_m)

    n₁, n₂, n₃, n₄, n₅ = size(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, n₁)
    j⁻ = j⁻ + 1
    j⁺ = min(j⁺ + 1, n₂)
    k⁻ = k⁻ + 1
    k⁺ = min(k⁺ + 1, n₃)
    l⁻ = l⁻ + 1
    l⁺ = min(l⁺ + 1, n₄)
    m⁻ = m⁻ + 1
    m⁺ = min(m⁺ + 1, n₅)

    return _interpolate(f.table, (i⁻, i⁺, ξ), (j⁻, j⁺, η), (k⁻, k⁺, ζ), (l⁻, l⁺, θ), (m⁻, m⁺, ψ))
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
