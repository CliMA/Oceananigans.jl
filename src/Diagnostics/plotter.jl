abstract type AbstractPlotter end

struct MovieMaker{F, FN, IO, NM} <: AbstractPlotter
    fig      :: F
    func     :: FN
    io       :: IO
    filename :: NM
end

function MovieMaker end

(maker::MovieMaker)(simulation) = maker.func(simulation, maker.fig, maker.io)
