import Oceananigans.Fields: broadcasted_to_abstract_operation

using Base.Broadcast: Broadcasted
using Base: identity

const BroadcastedIdentity = Broadcasted{<:Any, <:Any, typeof(identity), <:Any}

broadcasted_to_abstract_operation(loc, grid, bc::BroadcastedIdentity) =
    interpolate_operation(loc, Tuple(broadcasted_to_abstract_operation(loc, grid, a) for a in bc.args)...)

broadcasted_to_abstract_operation(loc, grid, op::AbstractOperation) = at(loc, op)

function broadcasted_to_abstract_operation(loc, grid, bc::Broadcasted{<:Any, <:Any, <:Any, <:Any})
    abstract_op = bc.f(loc, Tuple(broadcasted_to_abstract_operation(loc, grid, a) for a in bc.args)...)
    return interpolate_operation(loc, abstract_op) # For "stubborn" BinaryOperations
end
