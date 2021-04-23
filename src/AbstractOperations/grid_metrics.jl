using Oceananigans.Operators

# Define aliases for some spacings and areas at all locations.
for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)

    x_spacing_alias = Symbol(:Δx, LX, LY, LZ)
    y_spacing_alias = Symbol(:Δy, LX, LY, LZ)
    z_spacing_alias = Symbol(:Δz, LX, LY, LZ)

    z_area_alias = Symbol(:Az, LX, LY, LZ)

    x_spacing_function = Symbol(:Δx, LX, LY, :ᵃ)
    y_spacing_function = Symbol(:Δy, LX, LY, :ᵃ)
    z_spacing_function = Symbol(:Δz, :ᵃᵃ, LZ)

    z_area_function = Symbol(:Az, LX, LY, :ᵃ)

    @eval begin
        const $x_spacing_alias = $x_spacing_function
        const $y_spacing_alias = $y_spacing_function
        const $z_spacing_alias = $z_spacing_function
        const $z_area_alias = $z_area_function
    end
end

# Prototype functionality with x, y, z spacings
abstract type AbstractGridMetric end

struct XSpacingMetric <: AbstractGridMetric end 
struct YSpacingMetric <: AbstractGridMetric end 
struct ZSpacingMetric <: AbstractGridMetric end 

metric_function_prefix(::XSpacingMetric) = :Δx
metric_function_prefix(::YSpacingMetric) = :Δy
metric_function_prefix(::ZSpacingMetric) = :Δz

struct XAreaMetric <: AbstractGridMetric end 
struct YAreaMetric <: AbstractGridMetric end 
struct ZAreaMetric <: AbstractGridMetric end 

metric_function_prefix(::XAreaMetric) = :Ax
metric_function_prefix(::YAreaMetric) = :Ay
metric_function_prefix(::ZAreaMetric) = :Az

struct VolumeMetric <: AbstractGridMetric end 

metric_function_prefix(::VolumeMetric) = :V

# Convenient instances for users
const Δx = XSpacingMetric()
const Δy = YSpacingMetric()
const Δz = ZSpacingMetric()

const Ax = XAreaMetric()
const Ay = YAreaMetric()
const Az = ZAreaMetric()

const Volume = VolumeMetric()

"""
    metric_function(loc, metric::AbstractGridMetric)

Returns the function associated with `metric::AbstractGridMetric`
at `loc`ation.
"""
function metric_function(loc, metric::AbstractGridMetric)
    code = Tuple(interpolation_code(ℓ) for ℓ in loc)
    prefix = metric_function_prefix(metric)
    metric_function_symbol = Symbol(prefix, code...)
    return eval(metric_function_symbol)
end

struct GridMetricOperation{X, Y, Z, A, G, T, M} <: AbstractOperation{X, Y, Z, A, G, T}
          metric :: M
            grid :: G
    architecture :: A

    function GridMetricOperation{X, Y, Z}(metric::M, grid::G) where {X, Y, Z, M, G}
        arch = architecture(grid)
        A = typeof(arch)
        T = eltype(grid)
        return new{X, Y, Z, A, G, T, M}(metric, grid, arch)
    end
end

@inline Base.getindex(gm::GridMetricOperation, i, j, k) = gm.metric(i, j, k, gm.grid)

# Special constructor for BinaryOperation
GridMetricOperation(L, metric, grid) = GridMetricOperation{L[1], L[2], L[3]}(metric_function(L, metric), grid)

