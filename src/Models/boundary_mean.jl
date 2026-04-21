using Adapt, GPUArraysCore
using Oceananigans: instantiated_location
using Oceananigans.Fields: Center, Face
using Oceananigans.AbstractOperations: grid_metric_operation, Ax, Ay, Az
using Oceananigans.BoundaryConditions: BoundaryCondition, Open

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

    julia> set!(cf, (x, y, z) -> sin(2π * y / 4)); # hide output

    julia> bam = BoundaryAdjacentMean(grid, :east)
    BoundaryAdjacentMean: (0.0)

    julia> bam(:east, cf); # computes boundary-adjacent mean

    julia> abs(bam.value[]) < 1e-10  # essentially zero
    true
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

@inline boundary_normal_area(::Union{Val{:west}, Val{:east}}, grid)   = grid_metric_operation((Face, Center, Center), Ax, grid)
@inline boundary_normal_area(::Union{Val{:south}, Val{:north}}, grid) = grid_metric_operation((Center, Face, Center), Ay, grid)
@inline boundary_normal_area(::Union{Val{:bottom}, Val{:top}}, grid)  = grid_metric_operation((Center, Center, Face), Az, grid)

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

function boundary_total_area(::Val{:west}, grid)
    ∫dA = sum(boundary_normal_area(Val(:west), grid), dims=(2, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function boundary_total_area(::Val{:east}, grid)
    ∫dA = sum(boundary_normal_area(Val(:east), grid), dims=(2, 3))
    return @allowscalar ∫dA[grid.Nx+1, 1, 1]
end

function boundary_total_area(::Val{:south}, grid)
    ∫dA = sum(boundary_normal_area(Val(:south), grid), dims=(1, 3))
    return @allowscalar ∫dA[1, 1, 1]
end

function boundary_total_area(::Val{:north}, grid)
    ∫dA = sum(boundary_normal_area(Val(:north), grid), dims=(1, 3))
    return @allowscalar ∫dA[1, grid.Ny+1, 1]
end

function boundary_total_area(::Val{:bottom}, grid)
    ∫dA = sum(boundary_normal_area(Val(:bottom), grid), dims=(1, 2))
    return @allowscalar ∫dA[1, 1, 1]
end

function boundary_total_area(::Val{:top}, grid)
    ∫dA = sum(boundary_normal_area(Val(:top), grid), dims=(1, 2))
    return @allowscalar ∫dA[1, 1, grid.Nz+1]
end

boundary_total_area(side::Symbol, grid) = boundary_total_area(Val(side), grid)

"""
    LiveBoundaryTransport(velocity, side)

A callable that, when called with a `grid`, returns `velocity * boundary_total_area(side, grid)`.

Use this as the `target_mass_flux` argument of `PerturbationAdvection` when the target
transport should track the current grid geometry (e.g. with `ZStarCoordinate`, where column
height varies with the free surface).

# Example

```jldoctest
julia> using Oceananigans, Oceananigans.Models: LiveBoundaryTransport

julia> lbt = LiveBoundaryTransport(1.0, :east)
LiveBoundaryTransport: velocity=1.0, side=Val{:east}()

julia> grid = RectilinearGrid(size=(4, 1, 4), extent=(4, 1, 4));

julia> lbt(grid) ≈ 4.0  # 1.0 * (Ly * Lz) = 1 * 4
true
```
"""
struct LiveBoundaryTransport{FT, S}
    velocity :: FT
    side :: S
end

LiveBoundaryTransport(velocity::Number, side::Symbol) =
    LiveBoundaryTransport(velocity, Val(side))

(lbt::LiveBoundaryTransport)(grid) = lbt.velocity * boundary_total_area(lbt.side, grid)

Adapt.adapt_structure(to, lbt::LiveBoundaryTransport) = lbt

Base.show(io::IO, lbt::LiveBoundaryTransport) =
    print(io, "LiveBoundaryTransport: velocity=$(lbt.velocity), side=$(lbt.side)")
