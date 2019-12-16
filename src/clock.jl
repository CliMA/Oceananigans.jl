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

Base.show(io::IO, c::Clock{T}) where T =
    println(io, "Clock{$T}: time = ", prettytime(c.time), ", iteration = ", c.iteration)
