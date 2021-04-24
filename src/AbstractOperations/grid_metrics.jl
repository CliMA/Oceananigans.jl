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

"""
    Δz = ZSpacingMetric()

An instance of `AbstractGridMetric` that can be used to create
`BinaryOperation`s between `AbstractField`s and the vertical grid
spacing evaluated at the same location as the `AbstractField`. 

Example
=======

```julia
julia> using Oceananigans

julia> grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(2, 2, 2))
RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 2.0], y ∈ [0.0, 2.0], z ∈ [-2.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (1, 1, 1)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (2.0, 2.0, 2.0)

julia> c = CenterField(CPU(), grid);

julia> using Oceananigans.AbstractOperations: Δz

julia> c_dz = c * Δz
BinaryOperation at (Center, Center, Center)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
│   └── domain: x ∈ [0.0, 2.0], y ∈ [0.0, 2.0], z ∈ [-2.0, 0.0]
└── tree: 
    * at (Center, Center, Center)
    ├── Field located at (Center, Center, Center)
    └── Δzᵃᵃᶜ at (Center, Center, Center)

julia> set!(c, (x, y, z) -> rand())
1×1×1 view(OffsetArray(::Array{Float64,3}, 0:2, 0:2, 0:2), 1:1, 1:1, 1:1) with eltype Float64:
[:, :, 1] =
 0.1308337558599868

julia> c_dz[1, 1, 1]
0.2616675117199736
```
"""
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

