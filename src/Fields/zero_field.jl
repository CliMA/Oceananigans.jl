struct ZeroField <: AbstractField{Nothing, Nothing, Nothing, Nothing, Nothing, Nothing} end

@inline Base.getindex(::ZeroField, i, j, k) = 0
