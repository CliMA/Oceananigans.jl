"""
    DiscreteBoundaryFunction(func, parameters)

A wrapper for boundary condition functions with optional parameters.
When `parameters=nothing`, the boundary condition `func` is called with the signature

    `func(i, j, grid, clock, model_fields)`

where `i, j` are the indices along the boundary,
where `grid` is `model.grid`, `clock.time` is the current simulation time and
`clock.iteration` is the current model iteration, and
`model_fields` is a `NamedTuple` with `u, v, w`, the fields in `model.tracers`,
and the fields in `model.diffusivities`, each of which is an `OffsetArray`s (or `NamedTuple`s
of `OffsetArray`s depending on the turbulence closure) of field data.

When `parameters` is not `nothing`, the boundary condition `func` is called with
the signature

    `func(i, j, grid, clock, model_fields)`

*Note* that the index `end` does *not* access the final physical grid point of
a model field in any direction. The final grid point must be explictly specified, as
in `model_fields.u[i, j, grid.Nz]`*.

Example
=======

@inline linear_bottom_drag(i, j, grid, clock, model_fields, parameters) = 
    @inbounds - parameters.μ * model_fields.u[i, j, 1]

u_boundary_condition = BoundaryCondition(Flux, linear_bottom_drag,
                                         discrete_form=true, parameters=(μ=π,))
"""
struct DiscreteBoundaryFunction{P, F} <: Function
    func :: F
    parameters :: P
end

@inline (bc::DiscreteBoundaryFunction{Nothing})(args...) = bc.func(args...)
@inline (bc::DiscreteBoundaryFunction)(args...) = bc.func(args..., bc.parameters)

# Don't re-convert DiscreteBoundaryFunctions passed to BoundaryCondition constructor
BoundaryCondition(TBC, condition::DiscreteBoundaryFunction) =
    BoundaryCondition{TBC, typeof(condition)}(condition)
