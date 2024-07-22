include("dependencies_for_runtests.jl")

using Enzyme
using EnzymeCore
# Required presently
Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, DefaultBoundaryCondition, PeriodicBoundaryCondition, regularize_field_boundary_conditions, regularize_boundary_condition, RightBoundary

using Oceananigans.Operators: assumed_field_location

for dir in (:south, :north, :bottom, :top)
    extract_side_bc = Symbol(:extract_, dir, :_bc)
    @eval begin
        @inline $extract_side_bc(bc) = bc.$dir
        @inline $extract_side_bc(bc::Tuple) = map($extract_side_bc, bc)
    end
end

@inline extract_bc(bc, ::Val{:south_and_north}) = (extract_south_bc(bc), extract_north_bc(bc))
@inline extract_bc(bc, ::Val{:bottom_and_top})  = (extract_bottom_bc(bc), extract_top_bc(bc))

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions_low!(boundary_conditions, args...; kwargs...)
    
    sides = [:south_and_north, :bottom_and_top]
    perm  = [2,1]
    sides = sides[perm]

    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)
    number_of_tasks     = length(sides)

    return nothing
end

@testset "Enzyme on advection and diffusion WITH flux boundary condition" begin
    Nx = Ny = 64
    Nz = 8

    Lx = Ly = L = 2π
    Lz = 1

    x = y = (-L/2, L/2)
    z = (-Lz/2, Lz/2)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)

    @inline function tracer_flux(x, y, t, c)
        return c
    end

    top_c_bc = FluxBoundaryCondition(tracer_flux, field_dependencies=:c)

    loc = (Center, Center, nothing)
    top = regularize_boundary_condition(top_c_bc, grid, loc, 3, RightBoundary, tuple(:c))
    
    new_thing = FieldBoundaryConditions(; south = PeriodicBoundaryCondition(),
                                          north = PeriodicBoundaryCondition(),
                                          bottom = NoFluxBoundaryCondition(),
                                          top = top)

    dnew_thing = Enzyme.make_zero(new_thing)
    
    dc²_dκ = autodiff(Enzyme.Reverse,
                      fill_halo_regions_low!,
                      Duplicated(new_thing, dnew_thing),
                      Duplicated((1,2), (0,0)),
                      Duplicated((3,4), (0,0)))
    
    #=
    thing1 = Vector{Tuple{Tuple{BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification}, Tuple{BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition}}}
    thing2 = Int64
    thing3 = Vector{Tuple{Tuple{BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}}, Tuple{BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Oceananigans.BoundaryConditions.ContinuousBoundaryFunction{Center, Center, Nothing, Oceananigans.BoundaryConditions.RightBoundary, var"#tracer_flux#23", Nothing, Tuple{Symbol}, Tuple{Int64}, Tuple{typeof(Oceananigans.Operators.identity4)}}}}}}
    thing4 = Int64
    thing5 = Int64

    autodiff_thunk(ReverseSplitWithPrimal,
                Const{typeof(Base._unsafe_copyto!)},
                Active,
                Duplicated{thing1},
                Const{thing2},
                Duplicated{thing3},
                Const{thing4},
                Const{thing5})
    =#
end
