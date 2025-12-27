using Adapt

"""
    TabulatedFunction{F, T, FT}

A wrapper around a unary callable `func(x)` that precomputes values in a lookup table
for fast linear interpolation. This avoids expensive computations (like log, exp,
sqrt, atan, etc.) during time-critical operations like GPU kernel execution.

# Fields
- `func`: The original callable being tabulated (for reference/fallback)
- `table`: Precomputed values (`AbstractVector`)
- `x_min`: Minimum x value in table
- `x_max`: Maximum x value in table
- `inverse_Δx`: `1 / Δx` for fast index computation

# Example

```jldoctest tabulatedfunc
using Oceananigans

# Tabulate a simple function for fast evaluation
f = TabulatedFunction(sin; x_range=(0, 2π), points=1000)

# output
TabulatedFunction with 1000 points over [0.0, 6.283185307179586] of sin
```

```jldoctest tabulatedfunc
# Evaluate like a regular function
f(π/2)

# output
0.9999962914903766
```

See also [`tabulate`](@ref).
"""
struct TabulatedFunction{F, T, FT}
    func :: F
    table :: T
    x_min :: FT
    x_max :: FT
    inverse_Δx :: FT
end

"""
    TabulatedFunction(func; x_range=(-1, 1), points=1000, FT=Float64)

Construct a `TabulatedFunction` by precomputing `points` values of `func`
over the range `x_range` for fast linear interpolation.

# Arguments
- `func`: Any callable that takes a single numeric argument

# Keyword Arguments
- `x_range`: Tuple of `(minimum, maximum)` x values. Default: `(-1, 1)`
- `points`: Number of points in the lookup table. Default: `1000`
- `FT`: Float type for table values. Default: `Float64`

# Example

```jldoctest
using Oceananigans

# Tabulate a trigonometric function
f = TabulatedFunction(sin; x_range=(0, 2π), points=1000)

# Evaluate at π/4
f(π/4)

# output
0.7071052539107768
```

The tabulated function can be called like the original:

```julia
f(1.5)  # Returns interpolated value
```

Values outside `x_range` are clamped to the nearest table boundary.
"""
function TabulatedFunction(func, arch=CPU(), FT=Oceananigans.defaults.FloatType;
                           range,
                           points = 100)

    x_min, x_max = range
    Δx = (x_max - x_min) / (points - 1)
    inverse_Δx = 1 / Δx

    # Precompute table values
    table = [convert(FT, func(x_min + (i - 1) * Δx)) for i in 1:points]
    table = on_architecture(arch, table)

    return TabulatedFunction(func,
                             table,
                             convert(FT, x_min),
                             convert(FT, x_max),
                             convert(FT, inverse_Δx))
end

"""
    tabulate(func; x_range=(-1, 1), points=1000, FT=Float64)

Alias for `TabulatedFunction(func; ...)`. Creates a tabulated version
of `func` for fast evaluation via linear interpolation.

# Example

```jldoctest
using Oceananigans

# Tabulate an expensive computation
f = tabulate(x -> x^2 + exp(-x^2); x_range=(-5, 5), points=500)
f(2.0)

# output
4.01841070688188
```

See also [`TabulatedFunction`](@ref).
"""
tabulate(func; kwargs...) = TabulatedFunction(func; kwargs...)

#####
##### Evaluation via linear interpolation
#####

# Interpolator utility (returns 0-based indices and weight)
@inline function _tabulated_interpolator(fractional_idx)
    # For why we use Base.unsafe_trunc instead of trunc see:
    # https://github.com/CliMA/Oceananigans.jl/issues/828
    # https://github.com/CliMA/Oceananigans.jl/pull/997
    i⁻ = Base.unsafe_trunc(Int, fractional_idx)
    i⁺ = i⁻ + 1
    ξ = mod(fractional_idx, 1)
    return (i⁻, i⁺, ξ)
end

@inline function (f::TabulatedFunction)(x)
    # Clamp x to table range
    x_clamped = clamp(x, f.x_min, f.x_max)

    # Compute fractional index (uniform spacing)
    fractional_idx = (x_clamped - f.x_min) * f.inverse_Δx

    # Get interpolation indices and weight (0-based)
    i⁻, i⁺, ξ = _tabulated_interpolator(fractional_idx)

    # Convert to 1-based indices for Julia arrays and clamp upper bound
    n = length(f.table)
    i⁻ = i⁻ + 1
    i⁺ = min(i⁺ + 1, n)

    # Linear interpolation
    f⁻ = @inbounds f.table[i⁻]
    f⁺ = @inbounds f.table[i⁺]

    return (1 - ξ) * f⁻ + ξ * f⁺
end

#####
##### GPU/architecture support
#####

import Oceananigans.Architectures: on_architecture

on_architecture(arch, f::TabulatedFunction) =
    TabulatedFunction(f.func,
                      on_architecture(arch, f.table),
                      f.x_min,
                      f.x_max,
                      f.inverse_Δx)

# Adapt for GPU kernels (drops the original function to avoid GPU compilation issues)
Adapt.adapt_structure(to, f::TabulatedFunction) =
    TabulatedFunction(nothing,
                      Adapt.adapt(to, f.table),
                      f.x_min,
                      f.x_max,
                      f.inverse_Δx)

#####
##### Pretty printing
#####

function Base.summary(f::TabulatedFunction)
    n = length(f.table)
    return "TabulatedFunction with $(n) points over [$(f.x_min), $(f.x_max)]"
end

function Base.show(io::IO, f::TabulatedFunction)
    print(io, summary(f))
    if f.func !== nothing
        print(io, " of ", f.func)
    end
end

