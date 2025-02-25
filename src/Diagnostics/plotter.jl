abstract type AbstractPlotter end

struct MovieMaker{F, FN, IO, NM} <: AbstractPlotter
    fig      :: F
    func     :: FN
    io       :: IO
    filename :: NM
end

function MovieMaker end

function add_movie_maker!(simulation, schedule, args...; kwargs...)
    maker = MovieMaker(args...; kwargs...)
    add_callback!(simulation, maker, schedule)
end
