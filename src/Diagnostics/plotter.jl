import Adapt

abstract type AbstractPlotter end

struct MovieMaker{F, FN, IO, NM} <: AbstractPlotter
    figure   :: F
    func     :: FN
    io       :: IO
    filepath :: NM
end

function MovieMaker end

function add_movie_maker!(simulation, schedule, args...; kwargs...)
    maker = MovieMaker(args...; kwargs...)
    add_callback!(simulation, maker, schedule)
end

Base.summary(maker::MovieMaker) = string("MovieMaker saving to $(maker.filepath)")
function Base.show(io::IO, maker::MovieMaker)
    print(io, summary(maker), "\n")
    return print(io, "├── figure   = ", summary(maker.figure), "\n",
                     "├── func     = ", summary(maker.func), "\n",
                     "├── io       = ", summary(maker.io), "\n",
                     "└── filepath = ", maker.filepath)
end

function Adapt.adapt_structure(to, maker::MovieMaker)
    return MovieMaker(Adapt.adapt(to, maker.figure),
                      Adapt.adapt(to, maker.func),
                      Adapt.adapt(to, maker.io),
                      maker.filepath)
end
