import Base: show

"""
    Clock{T<:Number}

    Clock{T}(time, iteration)

Keeps track of the current `time` and `iteration` number. The `time::T` can be either a number of a `DateTime` object.
"""
mutable struct Clock{T}
       time :: T
  iteration :: Int
end

Clock(; time, iteration=0) = Clock(time, iteration)

Base.show(io::IO, c::Clock{FT}) where FT =
    println(io, "Clock{$FT}: time = ", prettytime(c.time), ", iteration = ", c.iteration)

function increment_clock!(clock, Δt)
    clock.time += Δt
    clock.iteration += 1
end

function increment_clock!(clock::Clock{DateTime}, Δt)
    clock.time += Nanosecond(round(Int, 1e9 * Δt))
    clock.iteration += 1
end
