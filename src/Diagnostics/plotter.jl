abstract type AbstractPlotter end

struct MovieMaker{F, FN, IO, NM} <: AbstractPlotter
    fig      :: F
    func     :: FN
    io       :: IO
    filename :: NM
end

MovieMaker() = throw(MethodError("Makie is needed."))

(maker::MovieMaker)(simulation) = maker.func(simulation, maker.fig, maker.io)
