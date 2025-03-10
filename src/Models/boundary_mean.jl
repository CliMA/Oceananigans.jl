using Adapt, CUDA
using Oceananigans: instantiated_location
using Oceananigans.Fields: Center, Face
using Oceananigans.AbstractOperations: GridMetricOperation, Ax, Ay, Az, @at
using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection

import Adapt: adapt_structure
import Base: summary, show
import Oceananigans.BoundaryConditions: update_boundary_condition!

"""
    BoundaryAdjacentMean

Stores the boundary mean `value` of a `Field`. The value is updated and stored
when the object is called. This automatically happens during 
`update_boundary_condition!` when `BoundaryAdjacentMean` is used as the condition
value in a boundary condition.

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.Models: BoundaryAdjacentMean

julia> grid = RectilinearGrid(topology = (Bounded, Periodic, Bounded), size = (16, 16, 16), extent = (3, 4, 5))
16×16×16 RectilinearGrid{Float64, Bounded, Periodic, Bounded} on CPU with 3×3×3 halo
├── Bounded  x ∈ [0.0, 3.0]  regularly spaced with Δx=0.1875
├── Periodic y ∈ [0.0, 4.0)  regularly spaced with Δy=0.25
└── Bounded  z ∈ [-5.0, 0.0] regularly spaced with Δz=0.3125

julia> bam = BoundaryAdjacentMean(grid, :east)
BoundaryAdjacentMean: (0.0)

julia> cf = CenterField(grid)
16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Bounded, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: ZeroFlux, east: ZeroFlux, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=0.0, min=0.0, mean=0.0

julia> set!(cf, (x, y, z) -> sin(2π * y / 4))
16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Bounded, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: ZeroFlux, east: ZeroFlux, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=0.980785, min=-0.980785, mean=1.11077e-16

julia> bam(:east, cf)
-7.806255641895632e-19

julia> ff = Field{Face, Center, Center}(grid)
17×16×16 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Bounded, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Nothing, east: Nothing, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 23×22×22 OffsetArray(::Array{Float64, 3}, -2:20, -2:19, -2:19) with eltype Float64 with indices -2:20×-2:19×-2:19
    └── max=0.0, min=0.0, mean=0.0

julia> set!(ff, (x, y, z) -> sin(2π * y / 4))
17×16×16 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Bounded, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Nothing, east: Nothing, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 23×22×22 OffsetArray(::Array{Float64, 3}, -2:20, -2:19, -2:19) with eltype Float64 with indices -2:20×-2:19×-2:19
    └── max=0.980785, min=-0.980785, mean=1.0284e-16

julia> bam(:east, ff)
-1.5612511283791264e-18

julia> using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition

julia> obc = PerturbationAdvectionOpenBoundaryCondition(bam)
┌ Warning: `PerturbationAdvection` open boundaries matching scheme is experimental and un-tested/validated
└ @ Oceananigans.BoundaryConditions ~/Oceananigans.jl/src/BoundaryConditions/perturbation_advection_open_boundary_matching_scheme.jl:52
OpenBoundaryCondition{Oceananigans.BoundaryConditions.PerturbationAdvection{Val{true}, Float64}}: BoundaryAdjacentMean: (-1.5612511283791264e-18)

julia> u_bcs = FieldBoundaryConditions(east = obc)
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── east: OpenBoundaryCondition{Oceananigans.BoundaryConditions.PerturbationAdvection{Val{true}, Float64}}: BoundaryAdjacentMean: (-1.5612511283791264e-18)
├── south: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── north: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── bottom: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── top: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
└── immersed: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)

```
"""
struct BoundaryAdjacentMean{NI, BV}
    boundary_normal_integral :: NI
                       value :: BV

   BoundaryAdjacentMean(grid, side; 
                        boundary_normal_integral::NI = boundary_reduced_field(Val(side), grid),
                        value::BV = Ref(zero(grid))) where {NI, BV} = 
        new{NI, BV}(boundary_normal_integral, value)
end

@inline (bam::BoundaryAdjacentMean)(args...) = bam.value[]

Adapt.adapt_structure(to, mo::BoundaryAdjacentMean) = 
    BoundaryAdjacentMean(; boundary_normal_integral = nothing, value = adapt(to, mo.value[]))

Base.show(io::IO, bam::BoundaryAdjacentMean) = print(io, summary(bam)*"\n")
Base.summary(bam::BoundaryAdjacentMean) = "BoundaryAdjacentMean: ($(bam.value[]))"

@inline boundary_reduced_field(::Union{Val{:west}, Val{:east}}, grid)   = Field{Face, Nothing, Nothing}(grid)
@inline boundary_reduced_field(::Union{Val{:south}, Val{:north}}, grid) = Field{Nothing, Face, Nothing}(grid)
@inline boundary_reduced_field(::Union{Val{:bottom}, Val{:top}}, grid)  = Field{Nothing, Nothing, Face}(grid)

@inline boundary_normal_area(::Union{Val{:west}, Val{:east}}, grid)   = GridMetricOperation((Face, Center, Center), Ax, grid)
@inline boundary_normal_area(::Union{Val{:south}, Val{:north}}, grid) = GridMetricOperation((Center, Face, Center), Ay, grid)
@inline boundary_normal_area(::Union{Val{:bottom}, Val{:top}}, grid)  = GridMetricOperation((Center, Center, Face), Az, grid)

@inline boundary_adjacent_indices(::Val{:east}, grid, loc) = size(grid, 1)+1, 1, 1
@inline boundary_adjacent_indices(val_side::Val{:west}, grid, loc) = 1, 1, 1

@inline boundary_adjacent_indices(::Val{:north}, grid, loc) = 1, size(grid, 2)+1, 1
@inline boundary_adjacent_indices(val_side::Val{:south}, grid, loc) = 1, 1, 1

@inline boundary_adjacent_indices(::Val{:top}, grid, loc) = 1, 1, size(grid, 3)+1
@inline boundary_adjacent_indices(val_side::Val{:bottom}, grid, loc) = 1, 1, 1

(bam::BoundaryAdjacentMean)(side, u) = bam(Val(side), u)

# computes the boundary mean and stores/returns it
function (bam::BoundaryAdjacentMean)(val_side::Val, u)
    grid = u.grid

    loc = instantiated_location(u)
    iB, jB, kB = boundary_adjacent_indices(val_side, grid, loc)
    An = boundary_normal_area(val_side, grid)

    # get the total flux
    sum!(bam.boundary_normal_integral, (@at (Face, Center, Center) u) * An)

    bam.value[] = CUDA.@allowscalar bam.boundary_normal_integral[iB, jB, kB]

    # get the normalizing area
    sum!(bam.boundary_normal_integral, An)

    bam.value[] /= CUDA.@allowscalar bam.boundary_normal_integral[iB, jB, kB]

    return bam.value[]
end

# let this get updated in boundary conditions

const MOOBC = BoundaryCondition{<:Open, <:BoundaryAdjacentMean}

@inline function update_boundary_condition!(bc::MOOBC, val_side, u, model)
    bc.condition(val_side, u)

    return nothing
end
