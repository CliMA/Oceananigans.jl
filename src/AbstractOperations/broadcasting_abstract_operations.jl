using Oceananigans.Fields: broadcasted_to_abstract_operation

using Base: identity
using Base.Broadcast: Broadcasted

const BroadcastedIdentity = Broadcasted{<:Any, <:Any, typeof(identity), <:Any}

@inline function Fields.broadcasted_to_abstract_operation(loc, grid, bc::BroadcastedIdentity)
    Nargs = length(bc.args)

    bc′ = ntuple(Val(Nargs)) do n
        broadcasted_to_abstract_operation(loc, grid, bc.args[n])
    end

    return interpolate_operation(loc, bc′...)
end

@inline Fields.broadcasted_to_abstract_operation(loc, grid, op::AbstractOperation) = at(loc, op)

@inline function Fields.broadcasted_to_abstract_operation(loc, grid, bc::Broadcasted{<:Any, <:Any, <:Any, <:Any})
    abstract_op = bc.f(loc, Tuple(broadcasted_to_abstract_operation(loc, grid, a) for a in bc.args)...)
    return interpolate_operation(loc, abstract_op) # For "stubborn" BinaryOperations
end
