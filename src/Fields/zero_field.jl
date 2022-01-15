struct ZeroField{N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, Nothing, N} end

ZeroField() = ZeroField{3}() # default

@inline Base.getindex(::ZeroField, ind...) = 0

