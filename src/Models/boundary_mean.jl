using Adapt, GPUArraysCore
using Oceananigans: instantiated_location
using Oceananigans.Fields: Center, Face
using Oceananigans.AbstractOperations: GridMetricOperation, Ax, Ay, Az
using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection

import Adapt: adapt_structure
import Base: summary, show
import Oceananigans.BoundaryConditions: update_boundary_condition!

struct BoundaryAdjacentMean{FF, BV}
    flux_field :: FF
         value :: BV

    @doc """
        BoundaryAdjacentMean(grid, side;
                             flux_field::FF = boundary_reduced_field(Val(side), grid),
                             value::BV = Ref(zero(grid)))

    Store the boundary mean `value` of a `Field`. Updated by calling

    ```jldoctest
    julia> using Oceananigans

    julia> using Oceananigans.Models: BoundaryAdjacentMean

    julia> grid = RectilinearGrid(size = (16, 16, 16), extent = (3, 4, 5));

    julia> cf = CenterField(grid);

    julia> set!(cf, (x, y, z) -> sin(2π * y / 4))
    16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
    ├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
    ├── boundary conditions: FieldBoundaryConditions
    │   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
    └── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
        └── max=0.980785, min=-0.980785, mean=-5.52808e-17

    julia> bam = BoundaryAdjacentMean(grid, :east)
    BoundaryAdjacentMean: (0.0)

    julia> bam(:east, cf)

    julia> bam
    BoundaryAdjacentMean: (-1.5612511283791264e-18)
    ```
    """
    BoundaryAdjacentMean(grid, side;
                         flux_field::FF = boundary_reduced_field(Val(side), grid),
                         value::BV = Ref(zero(grid))) where {FF, BV} =
        new{FF, BV}(flux_field, value)
end

@inline (bam::BoundaryAdjacentMean)(args...) = bam.value[]

Adapt.adapt_structure(to, mo::BoundaryAdjacentMean) =
    BoundaryAdjacentMean(; flux_fields = nothing, value = adapt(to, mo.value[]))

Base.show(io::IO, bam::BoundaryAdjacentMean) = print(io, summary(bam))
Base.summary(bam::BoundaryAdjacentMean) = "BoundaryAdjacentMean: ($(bam.value[]))"

@inline boundary_reduced_field(::Union{Val{:west}, Val{:east}}, grid)   = Field{Center, Nothing, Nothing}(grid)
@inline boundary_reduced_field(::Union{Val{:south}, Val{:north}}, grid) = Field{Nothing, Center, Nothing}(grid)
@inline boundary_reduced_field(::Union{Val{:bottom}, Val{:top}}, grid)  = Field{Nothing, Nothing, Center}(grid)

@inline boundary_normal_area(::Union{Val{:west}, Val{:east}}, grid)   = GridMetricOperation((Face, Center, Center), Ax, grid)
@inline boundary_normal_area(::Union{Val{:south}, Val{:north}}, grid) = GridMetricOperation((Center, Face, Center), Ay, grid)
@inline boundary_normal_area(::Union{Val{:bottom}, Val{:top}}, grid)  = GridMetricOperation((Center, Center, Face), Az, grid)

@inline boundary_adjacent_indices(::Val{:east}, grid, loc) = size(grid, 1), 1, 1
@inline boundary_adjacent_indices(val_side::Val{:west}, grid, loc) = first_interior_index(val_side, loc), 1, 1

@inline boundary_adjacent_indices(::Val{:north}, grid, loc) = 1, size(grid, 2), 1
@inline boundary_adjacent_indices(val_side::Val{:south}, grid, loc) = 1, first_interior_index(val_side, loc), 1

@inline boundary_adjacent_indices(::Val{:top}, grid, loc) = 1, 1, size(grid, 3)
@inline boundary_adjacent_indices(val_side::Val{:bottom}, grid, loc) = 1, 1, first_interior_index(val_side, loc)

@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{Center, <:Any, <:Any}) = 1
@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{Face, <:Any, <:Any}) = 2

@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, Center, <:Any}) = 1
@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, Face, <:Any}) = 2

@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, Center}) = 1
@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, Face}) = 2

(bam::BoundaryAdjacentMean)(side, u) = bam(Val(side), u)

# Computes the boundary mean and stores it.
function (bam::BoundaryAdjacentMean)(val_side::Val, u)
    grid = u.grid

    loc = instantiated_location(u)
    iB, jB, kB = boundary_adjacent_indices(val_side, grid, loc)
    An = boundary_normal_area(val_side, grid)

    # get the total flux
    sum!(bam.flux_field, u * An)

    bam.value[] = @allowscalar bam.flux_field[iB, jB, kB]

    # get the normalizing area
    sum!(bam.flux_field, An)

    bam.value[] /= @allowscalar bam.flux_field[iB, jB, kB]

    return nothing
end

const MOOBC = BoundaryCondition{<:Open, <:BoundaryAdjacentMean}
@inline update_boundary_condition!(bc::MOOBC, val_side, u, model) = bc.condition(val_side, u)
