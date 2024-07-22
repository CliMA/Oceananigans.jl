include("dependencies_for_runtests.jl")

using Enzyme
using EnzymeCore
# Required presently
Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

mutable struct FieldBoundaryConditions{S, N, B, T}
       south :: S
       north :: N
      bottom :: B
         top :: T
end

struct ContinuousBoundaryFunction{X, Y, Z, F, P, D, N}
                          func :: F
                    parameters :: P
            field_dependencies :: D
    field_dependencies_indices :: N

    function ContinuousBoundaryFunction{X, Y, Z}(func::F,
                                                    parameters::P,
                                                    field_dependencies::D,
                                                    field_dependencies_indices::N) where {X, Y, Z, F, P, D, N}

        return new{X, Y, Z, F, P, D, N}(func, parameters, field_dependencies, field_dependencies_indices)
    end
end

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

    @inline function tracer_flux(x, y, t, c)
        return c
    end

    regularized_boundary_func = ContinuousBoundaryFunction{Center, Center, nothing}(tracer_flux,
                                            nothing,
                                            tuple(:c),
                                            (1,))

    new_thing = FieldBoundaryConditions((1,), (1,), (2,), tuple(regularized_boundary_func))

    dnew_thing = Enzyme.make_zero(new_thing)
    
    dc²_dκ = autodiff(Enzyme.Reverse,
                      fill_halo_regions_low!,
                      Duplicated(new_thing, dnew_thing),
                      Duplicated((1,2), (0,0)),
                      Duplicated((3,4), (0,0)))
end
