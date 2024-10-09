
abstract type AbstractStaticCoordinate end

struct FlatCoordinate{C} <: AbstractStaticCoordinate 
    c :: C
end

struct Static1DCoordinate{D, C} <: AbstractStaticCoordinate
    cᶠ :: C
    cᶜ :: C
    Δᶠ :: D
    Δᶜ :: D
end

const UniformCoordinate = Static1DCoordinate{<:Number}
const StretechedCoordinate = Static1DCoordinate{<:Number}

const C = Center
const F = Face

coordinate(i, c::StaticRectilinearCoordinate, ::C) = c.cᶠ[i]
coordinate(i, c::StaticRectilinearCoordinate, ::F) = c.cᶠ[i]

spacing(i, c::StaticRectilinearCoordinate, ::C) = c.Δᶜ[i]
spacing(i, c::StaticRectilinearCoordinate, ::F) = c.Δᶠ[i]
