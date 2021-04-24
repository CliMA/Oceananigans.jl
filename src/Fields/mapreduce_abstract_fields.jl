using GPUArrays

using Oceananigans.Architectures: AbstractGPUArchitecture

#####
##### In place reductions!
#####

for (function_name, reduction_operation) in ((:sum,     :(Base.add_sum)),
                                             (:prod,    :(Base.mul_prod)),
                                             (:maximum, :(Base.max)),
                                             (:minimum, :(Base.min)),
                                             (:all,     :&),
                                             (:any,     :|))

    function_name! = Symbol(function_name, '!')

    @eval begin

        function Base.$(function_name!)(f::Function,
                                        result::AbstractReducedField{LX, LY, LZ, <:AbstractGPUArchitecture, G, T},
                                        operand::AbstractArray{T}) where {LX, LY, LZ, G, T}

            return Base.mapreducedim!(f, $(reduction_operation), interior(result), operand;
                                      init = GPUArrays.neutral_element($(reduction_operation), T))
        end

    end
end
