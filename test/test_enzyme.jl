include("dependencies_for_runtests.jl")

using Enzyme
using EnzymeCore
# Required presently
Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

mutable struct OuterStruct{S, T}
    a:: S
    b:: T
end

struct InnerStruct{F}
    func :: F

    function InnerStruct(func::F) where {F}
        return new{F}(func)
    end
end

for dir in (:a, :b)
    extract_side = Symbol(:extract_, dir)
    @eval begin
        @inline $extract_side(bc) = bc.$dir
        @inline $extract_side(bc::Tuple) = map($extract_side, bc)
    end
end

@inline extract(bc, ::Val{:a_and_a}) = (extract_a(bc), extract_a(bc))
@inline extract(bc, ::Val{:a_and_b})  = (extract_a(bc), extract_b(bc))

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function tuple_things!(inner, args...; kwargs...)
    
    sides = [:a_and_a, :a_and_b]
    perm  = [2,1]
    sides = sides[perm]

    inner = Tuple(extract(inner, Val(side)) for side in sides)
    number_of_tasks = length(sides)

    return nothing
end

@testset "Enzyme on advection and diffusion WITH flux boundary condition" begin

    b = InnerStruct(1.0)
    thing  = OuterStruct((1,), tuple(b))
    dthing = Enzyme.make_zero(thing)
    
    dc²_dκ = autodiff(Enzyme.Reverse,
                      tuple_things!,
                      Duplicated(thing, dthing),
                      Duplicated((1,2), (0,0)),
                      Duplicated((3,4), (0,0)))
end
