struct ExtremaPreservation{E, R, I, FT}
    extrema :: E
    Δτ :: R
    initialized :: I
    downwind_weight :: FT
end

"""
    ExtremaPreservation(; downwind_weight = 7//10)

Limit every face flux of a tracer in one pass so that its advective update stays within the extrema `(m, M)` of the tracer.

The update of a cell is the mean of its `n` one-direction updates `c + n κ D`, with `κ = Δτ / V`, so each direction stays within the
extrema on its own. A face with volume flux `q`, donor value `c_D` and reconstruction `ψ` carries `q (c_D + θ (ψ - c_D))`, and `θ` is the
smaller of `min(1, Q / P)` over its two cells, so that both compute the same flux. For a cell that the antidiffusive flux
`A = q (ψ - c_D)` lowers, `Q = cᵘ - m` is the distance from its upwind update `cᵘ` to the minimum (`M - cᵘ` when `A` raises it) and
`P = n κ (|A| + Aᵒ)`, where `Aᵒ` is the correction of the cell's other face along the direction, whose value is estimated at
`c_D + α (c_R - c_D)` with `α = downwind_weight` (0 is upwind, 1/2 centered, 1 downwind), and which moves the cell the same way whether
the flow enters or leaves it.

The extrema are the range of the point values of the tracer at the first time step, each cell value widened by half its largest minmod
slope. For one cell along one direction they widen by the cell's own value and the values that flow into it, so what a source or a boundary
flux puts beyond the extrema is advected. A smooth extremum beyond them also rises by 1/8 of the smaller second difference of the two cells
of the face, the rise of a cell mean as the extremum of a parabola reaches its center, and by at most 1/8 of how far both cells are beyond
the extrema, which leaves out a step and a face next to a wall.

The extrema are one pair (minimum, maximum) in a one-element array on the architecture of the grid; the time step `Δτ` of the update the
next tendency enters and the flag `initialized` are `Ref`s on the host. All are `nothing` until the scheme is materialized on a grid.
"""
function ExtremaPreservation(; downwind_weight = 7//10)
    0 ≤ downwind_weight ≤ 1 || throw(ArgumentError("downwind_weight must lie in [0, 1], found $downwind_weight"))
    return ExtremaPreservation(nothing, nothing, nothing, downwind_weight)
end

Base.show(io::IO, bounds::ExtremaPreservation) = print(io, "ExtremaPreservation(downwind_weight=", bounds.downwind_weight, ")")

Adapt.adapt_structure(to, bounds::ExtremaPreservation) =
    ExtremaPreservation(Adapt.adapt(to, bounds.extrema), _unwrap_for_gpu(bounds.Δτ), nothing, bounds.downwind_weight)

const ExtremaPreservingWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:ExtremaPreservation}
