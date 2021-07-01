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

#=
# For the reason why this dispatch is needed, see: https://github.com/CliMA/Oceananigans.jl/issues/1767
function Statistics.mean!(R::AbstractReducedField, A::AbstractArray)
    sum!(R, A; init=true)
    x = max(1, length(R)) // length(A)
    parent(R) .= parent(R) .* x 
    return R
end
=#
