import Base: show

using Dates: AbstractTime, Nanosecond

"""
    Clock{T<:Number}


Keeps track of the current `time` and `iteration` number. The `time::T` can be either a number of a `DateTime` object.
"""
mutable struct Clock{T}
         time :: T
    iteration :: Int
        stage :: Int
end

"""
    Clock(time, iteration=0, stage=1)

Returns an initialized `Clock`.
"""
Clock(; time, iteration=0, stage=1) = Clock(time, iteration, stage)

Base.show(io::IO, c::Clock{T}) where T =
    println(io, "Clock{$T}: time = ", prettytime(c.time),
                    ", iteration = ", c.iteration,
                        ", stage = ", c.stage)

tick_time!(clock, Δt) = clock.time += Δt
tick_time!(clock::Clock{<:AbstractTime}, Δt) = clock.time += Nanosecond(round(Int, 1e9 * Δt))
    
function tick!(clock, Δt; stage=false)

    tick_time!(clock, Δt)

    if stage # tick a stage update
        clock.stage += 1
    else # tick an iteration and reset stage
        clock.iteration += 1
        clock.stage = 1
    end

    return nothing
end

"Adapt `Clock` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, clock::Clock) = (time=clock.time, iteration=clock.iteration, stage=clock.stage)
