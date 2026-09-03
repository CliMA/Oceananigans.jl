"""
    BoundsPreservation(minimum_value, maximum_value, maximum_courant_number, limiter)

The interval a tracer is restricted to, the Courant number ``ω̂₁`` up to which the bound holds, and the field
`limiter` holding the cell-wise rescaling factor ``θ``. `limiter` is `nothing` until the scheme is materialized
on a grid.
"""
struct BoundsPreservation{FT, L}
    minimum_value :: FT
    maximum_value :: FT
    maximum_courant_number :: FT
    limiter :: L
end

Base.show(io::IO, bounds::BoundsPreservation) = print(io, "(", bounds.minimum_value, ", ", bounds.maximum_value, ")")

Adapt.adapt_structure(to, bounds::BoundsPreservation) =
    BoundsPreservation(Adapt.adapt(to, bounds.minimum_value),
                       Adapt.adapt(to, bounds.maximum_value),
                       Adapt.adapt(to, bounds.maximum_courant_number),
                       Adapt.adapt(to, bounds.limiter))

Architectures.on_architecture(to, bounds::BoundsPreservation) =
    BoundsPreservation(on_architecture(to, bounds.minimum_value),
                       on_architecture(to, bounds.maximum_value),
                       on_architecture(to, bounds.maximum_courant_number),
                       on_architecture(to, bounds.limiter))
