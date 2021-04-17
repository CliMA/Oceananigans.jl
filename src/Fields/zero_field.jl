struct ZeroField{X, Y, Z} <: AbstractField{X, Y, Z, Nothing, Nothing, Nothing} end

@inline Base.getindex(::ZeroField, i, j, k) = 0
