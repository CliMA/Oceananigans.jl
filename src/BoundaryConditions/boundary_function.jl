"""
        BoundaryFunction{B, X1, X2}(func, parameters=nothing)

A wrapper for the user-defined boundary condition function `func`, on the
boundary specified by symbol `B` and at location `(X1, X2)`, and with `parameters`.

Example
=======

julia> using Oceananigans, Oceananigans.BoundaryConditions, Oceananigans.Fields

julia> top_tracer_flux = BoundaryFunction{:z, Cell, Cell}((x, y, t) -> cos(2π*x) * cos(t))
(::BoundaryFunction{:z,Cell,Cell,var"#7#8",Nothing}) (generic function with 1 method)

julia> top_tracer_bc = BoundaryCondition(Flux, top_tracer_flux);

julia> flux_func(x, y, t, p) = cos(p.k * x) * cos(p.ω * t); # function with parameters

julia> parameterized_u_velocity_flux = BoundaryFunction{:z, Face, Cell}(flux_func, (k=4π, ω=3.0))
(::BoundaryFunction{:z,Face,Cell,typeof(flux_func),NamedTuple{(:k, :ω),Tuple{Float64,Float64}}}) (generic function with 1 method)

julia> top_u_bc = BoundaryCondition(Flux, parameterized_u_velocity_flux);
"""
struct BoundaryFunction{B, X1, X2, F, P} <: Function
          func :: F
    parameters :: P

    function BoundaryFunction{B, X1, X2}(func; parameters=nothing) where {B, X1, X2}

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
