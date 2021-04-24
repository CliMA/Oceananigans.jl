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

            ii = interior_parent_indices(location(result, 1), topology(result.grid, 1), result.grid.Nx, result.grid.Hx)
            ji = interior_parent_indices(location(result, 2), topology(result.grid, 2), result.grid.Ny, result.grid.Hy)
            ki = interior_parent_indices(location(result, 3), topology(result.grid, 3), result.grid.Nz, result.grid.Hz)

            result_interior = view(result, ii, ji, ki)

            return Base.mapreducedim!(f, $(reduction_operation), result_interior, operand;
                                      init = GPUArrays.neutral_element($(reduction_operation), T))
        end

    end
end
