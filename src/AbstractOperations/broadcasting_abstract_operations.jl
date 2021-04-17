using Oceananigans.Fields: FieldBroadcastStyle

using Base.Broadcast: Broadcasted

import Base.Broadcast: broadcasted

for op in binary_operators
    O = typeof(op)
    @eval broadcasted(::FieldBroadcastStyle, op::$O, a, b) = op(a, b)
end

for op in multiary_operators
    O = typeof(op)
    @eval broadcasted(::FieldBroadcastStyle, op::$O, a, b, c, d...) = op(a, b, c, d...)
end
