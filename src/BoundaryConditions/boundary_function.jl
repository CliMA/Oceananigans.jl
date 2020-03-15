"""
        BoundaryFunction{B, X1, X2}(func, parameters=nothing)

A wrapper for user-defined boundary condition functions on the
boundary specified by symbol `B` and at location `(X1, X2)`.

Example
=======
julia> using Oceananigans: BoundaryCondition, BoundaryFunction, Flux, Cell

julia> top_tracer_flux = BoundaryFunction{:z, Cell, Cell}((x, y, t) -> cos(2π*x) * cos(t))
(::BoundaryFunction{:z,Cell,Cell,getfield(Main, Symbol("##7#8"))}) (generic function with 1 method)

julia> top_tracer_bc = BoundaryCondition(Flux, top_tracer_flux);

julia> momentum_flux_func(x, y, t, p) = cos(p.k * x) * cos(p.ω * t);

julia> parameterized_u_velocity_flux = BoundaryFunction{:z, Face, Cell}(flux_func, (k=4π, ω=3.0))

julia> top_u_bc = BoundaryCondition(Flux, parameterized_u_velocity_flux);
"""
struct BoundaryFunction{B, X1, X2, F, P} <: Function
    func :: F
    parameters :: P

    function BoundaryFunction{B, X1, X2}(func, parameters=nothing) where {B, X1, X2}
        B ∈ (:x, :y, :z) || throw(ArgumentError("The boundary B at which the BoundaryFunction is
                                                to be applied must be either :x, :y, or :z."))
        return new{B, X1, X2, typeof(func), typeof(parameters)}(func, parameters)
    end
end

@inline call_boundary_function(func, ξ, η, t, ::Nothing) = func(ξ, η, t)
@inline call_boundary_function(func, ξ, η, t, parameters) = func(ξ, η, t, parameters)

@inline (bc::BoundaryFunction{:x, Y, Z})(j, k, grid, clock, state) where {Y, Z} =
    call_boundary_function(bc.func, ynode(Y, j, grid), znode(Z, k, grid), clock.time, bc.parameters)

@inline (bc::BoundaryFunction{:y, X, Z})(i, k, grid, clock, state) where {X, Z} =
    call_boundary_function(bc.func, xnode(X, i, grid), znode(Z, k, grid), clock.time, bc.parameters)

@inline (bc::BoundaryFunction{:z, X, Y})(i, j, grid, clock, state) where {X, Y} =
    call_boundary_function(bc.func, xnode(X, i, grid), ynode(Y, j, grid), clock.time, bc.parameters)

#####
##### Convenience constructors
#####

""" Returns the location of `loc` on the boundary `:x`, `:y`, or `:z`. """
BoundaryLocation(::Val{:x}, loc) = (loc[2], loc[3])
BoundaryLocation(::Val{:y}, loc) = (loc[1], loc[3])
BoundaryLocation(::Val{:z}, loc) = (loc[1], loc[2])

""" Returns a boundary function at on the boundary `B` at the appropriate
    location for a tracer field. """
TracerBoundaryFunction(B, args...) = BoundaryFunction{B, Cell, Cell}(args...)

""" Returns a boundary function at on the boundary `B` at the appropriate
    location for u, the x-velocity field. """
function UVelocityBoundaryFunction(B, args...)
    loc = BoundaryLocation(Val(B), (Face, Cell, Cell))
    return BoundaryFunction{B, loc[1], loc[2]}(args...)
end

""" Returns a boundary function at on the boundary `B` at the appropriate
    location for v, the y-velocity field. """
function VVelocityBoundaryFunction(B, args...)
    loc = BoundaryLocation(Val(B), (Cell, Face, Cell))
    return BoundaryFunction{B, loc[1], loc[2]}(args...)
end

""" Returns a boundary function at on the boundary `B` at the appropriate
    location for w, the z-velocity field. """
function WVelocityBoundaryFunction(B, args...)
    loc = BoundaryLocation(Val(B), (Cell, Cell, Face))
    return BoundaryFunction{B, loc[1], loc[2]}(args...)
end

"""
    TracerBoundaryCondition(bctype, B, args...)

Returns a `BoundaryCondition` of type `bctype`, that applies the function
`func` to a tracer on the boundary `B`, which is one of `:x, :y, :z`.
The boundary function has the signature

    `func(ξ, η, t)`

where `t` is time, and `ξ` and `η` are coordinates along the
boundary, eg: `(y, z)` for `B = :x`, `(x, z)` for `B = :y`, or
`(x, y)` for `B = :z`.
"""
TracerBoundaryCondition(bctype, B, args...) =
    BoundaryCondition(bctype, TracerBoundaryFunction(B, args...))

"""
    UVelocityBoundaryCondition(bctype, B, args...)

Returns a `BoundaryCondition` of type `bctype`, that applies the function
`func` to `u`, the `x`-velocity field, on the boundary `B`, which is one
of `:x, :y, :z`. The boundary function has the signature

    `func(ξ, η, t)`

where `t` is time, and `ξ` and `η` are coordinates along the
boundary, eg: `(y, z)` for `B = :x`, `(x, z)` for `B = :y`, or
`(x, y)` for `B = :z`.
"""
UVelocityBoundaryCondition(bctype, B, args...) =
    BoundaryCondition(bctype, UVelocityBoundaryFunction(B, args...))

"""
    VVelocityBoundaryCondition(bctype, B, args...)

Returns a `BoundaryCondition` of type `bctype`, that applies the function
`func` to `v`, the `y`-velocity field, on the boundary `B`, which is one
of `:x, :y, :z`. The boundary function has the signature

    `func(ξ, η, t)`

where `t` is time, and `ξ` and `η` are coordinates along the
boundary, eg: `(y, z)` for `B = :x`, `(x, z)` for `B = :y`, or
`(x, y)` for `B = :z`.
"""
VVelocityBoundaryCondition(bctype, B, args...) =
    BoundaryCondition(bctype, VVelocityBoundaryFunction(B, args...))

"""
    VVelocityBoundaryCondition(bctype, B, args...)

Returns a `BoundaryCondition` of type `bctype`, that applies the function
`func` to `w`, the `z`-velocity field, on the boundary `B`, which is one
of `:x, :y, :z`. The boundary function has the signature

    `func(ξ, η, t)`

where `t` is time, and `ξ` and `η` are coordinates along the
boundary, eg: `(y, z)` for `B = :x`, `(x, z)` for `B = :y`, or
`(x, y)` for `B = :z`.
"""
WVelocityBoundaryCondition(bctype, B, args...) =
    BoundaryCondition(bctype, WVelocityBoundaryFunction(B, args...))
