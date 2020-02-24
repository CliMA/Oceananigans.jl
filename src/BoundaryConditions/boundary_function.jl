"""
    BoundaryFunction{B, X1, X2, F}

A wrapper for user-defined boundary condition functions.
"""
struct BoundaryFunction{B, X1, X2, F} <: Function
    func :: F

    """
        BoundaryFunction{B, X1, X2}(func)

    A wrapper for user-defined boundary condition functions on the
    boundary specified by symbol `B` and at location `(X1, X2)`.

    Example
    =======
    julia> using Oceananigans: BoundaryCondition, BoundaryFunction, Flux, Cell

    julia> top_tracer_flux = BoundaryFunction{:z, Cell, Cell}((x, y, t) -> cos(2π*x) * cos(t))
    (::BoundaryFunction{:z,Cell,Cell,getfield(Main, Symbol("##7#8"))}) (generic function with 1 method)

    julia> top_tracer_bc = BoundaryCondition(Flux, top_tracer_flux);
    """
    function BoundaryFunction{B, X1, X2}(func) where {B, X1, X2}
        B ∈ (:x, :y, :z) || throw(ArgumentError("The boundary B at which the BoundaryFunction is
                                                to be applied must be either :x, :y, or :z."))
        new{B, X1, X2, typeof(func)}(func)
    end
end
