"""
    Clock{T<:Number}

    Clock{T}(time, iteration)

Keeps track of the current `time` and `iteration` number.
"""
mutable struct Clock{T<:Number}
       time :: T
  iteration :: Int
end
