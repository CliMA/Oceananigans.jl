using Statistics
using Oceananigans.Architectures: AbstractGPUArchitecture

#####
##### In place reductions!
#####

for function_name in (:sum, :prod, :maximum, :minimum, :all, :any)

    function_name! = Symbol(function_name, '!')

    # Unwrap ReducedField to a view over interior nodes:
    @eval begin
        Base.$(function_name!)(f::Function, r::AbstractReducedField, a::AbstractArray; kwargs...) = Base.$(function_name!)(f, interior(r), a; kwargs...)
        Base.$(function_name!)(r::AbstractReducedField, a::AbstractArray; kwargs...) = Base.$(function_name!)(identity, interior(r), a; kwargs...)
    end
end
