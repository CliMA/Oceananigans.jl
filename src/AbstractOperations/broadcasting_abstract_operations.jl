using Base.Broadcast: Broadcasted

using Oceananigans.Fields: FieldBroadcastStyle

import Base.Broadcast: broadcasted

import Oceananigans.Fields: insert_destination_location

for op in binary_operators
    O = typeof(eval(op))
    @eval broadcasted(::FieldBroadcastStyle, op::$O, a, b) = op(a, b)
end

for op in multiary_operators
    O = typeof(eval(op))
    @eval broadcasted(::FieldBroadcastStyle, op::$O, a, b, c, d...) = op(a, b, c, d...)
end

insert_destination_location(loc, op::AbstractOperation) = at(loc, bc)
