struct ExtremaPreservation{E, R, I, FT}
    extrema :: E
    Δτ :: R
    initialized :: I
    downwind_weight :: FT
end

"""
    ExtremaPreservation(; downwind_weight = 7//10)

Limit the face fluxes of a tracer in one pass so that its advective update stays within the extrema of the tracer. A face with flux `q`,
upwind value `c_D` and reconstruction `ψ` carries `q (c_D + θ (ψ - c_D))`, where `θ ∈ [0, 1]` is the largest fraction of the high-order
correction that keeps both cells of the face within the extrema.

The extrema are the range of the tracer at the first time step. For each cell they widen to the values that flow into it, so what sources
and boundary fluxes add is advected, and by the rise of a smooth extremum within its cell, so smooth peaks keep their height while steps
are clipped.

Each face sets `θ` before the other face of its cell is known, so it leaves room for that face's correction, estimated at the value
`c_D + α (c_R - c_D)` between that face's upwind and downwind values, with `α = downwind_weight` (0 upwind, 1/2 centered, 1 downwind).
A larger `α` leaves more room: fewer excursions beyond the extrema, but more limiting and more diffusion. A smaller `α` limits less, but
the two faces of a cell can together overshoot.
"""
function ExtremaPreservation(; downwind_weight = 7//10)
    0 ≤ downwind_weight ≤ 1 || throw(ArgumentError("downwind_weight must lie in [0, 1], found $downwind_weight"))
    return ExtremaPreservation(nothing, nothing, nothing, downwind_weight)
end

Base.show(io::IO, bounds::ExtremaPreservation) = print(io, "ExtremaPreservation(downwind_weight=", bounds.downwind_weight, ")")

Adapt.adapt_structure(to, bounds::ExtremaPreservation) =
    ExtremaPreservation(Adapt.adapt(to, bounds.extrema), _unwrap_for_gpu(bounds.Δτ), nothing, bounds.downwind_weight)

const ExtremaPreservingWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:ExtremaPreservation}
