abstract type AbstractPlotter end

struct MovieMaker{F, IO, FN, NM} <: AbstractPlotter
    fig      :: F
    io       :: IO
    func     :: FN
    filename :: NM
end

"""
    MovieMaker(func; fig, io=nothing, filename="movie.mp4", kwargs...)

Create a `MovieMaker`, which makes a movie.
"""
function MovieMaker(func; fig, io=nothing, filename="movie.mp4")
    return MovieMaker(fig, io, func, filename)
end

(maker::MovieMaker)(simulation) = maker.func(simulation, maker.fig, maker.io)
