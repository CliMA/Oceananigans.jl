module Utils

using Oceananigans
using Reactant

import Oceananigans.Utils: prettytime, prettysummary, heuristic_workgroup

function prettytime(concrete_number::Union{ConcretePJRTNumber,ConcreteIFRTNumber})
    number = Reactant.to_number(concrete_number)
    return prettytime(number)
end

function prettysummary(concrete_number::ConcretePJRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcretePJRTNumber(", prettysummary(number), ")")
end

function prettysummary(concrete_number::ConcreteIFRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcreteIFRTNumber(", prettysummary(number), ")")
end

#####
##### Reactant-specific workgroup heuristics
#####
##### On Reactant, we use larger workgroups to avoid nested affine.parallel + affine.if
##### structures that fail with raise=true. Using workgroup size ≥ worksize ensures
##### a single workgroup, avoiding the nested structure entirely.
#####

# const MAX_REACTANT_WORKGROUP = 256

# # 1D case
# heuristic_workgroup(::Oceananigans.ReactantState, Wx) = min(Wx, MAX_REACTANT_WORKGROUP)

# # 2D/3D/4D case - use workgroup that encompasses all work (up to limit)
# function heuristic_workgroup(::Oceananigans.ReactantState, Wx::Int, Wy::Int, Wz=nothing, Wt=nothing)
#     if Wx == 1 && Wy == 1
#         return (1, 1)
#     elseif Wx == 1
#         return (1, min(MAX_REACTANT_WORKGROUP, Wy))
#     elseif Wy == 1
#         return (min(MAX_REACTANT_WORKGROUP, Wx), 1)
#     else
#         # For Reactant: use workgroup size that encompasses the work to avoid
#         # nested affine.parallel loops that fail with raise=true
#         wg_x = min(Wx, MAX_REACTANT_WORKGROUP)
#         wg_y = min(Wy, MAX_REACTANT_WORKGROUP)
#         return (wg_x, wg_y)
#     end
# end

end # module
