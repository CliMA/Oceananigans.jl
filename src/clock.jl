import Base: show

"""
    Clock{T<:Number}

    Clock{T}(time, iteration)

Keeps track of the current `time` and `iteration` number.
"""
mutable struct Clock{T<:Number}
       time :: T
  iteration :: Int
end

Base.show(io::IO, c::Clock{FT}) where FT =
    println(io, "Clock{$FT}: time = ", prettytime(c.time), ", iteration = ", c.iteration)
