"""
    ParameterizedDiscreteBoundaryFunction(func, parameters)

A wrapper for boundary condition functions implemented as functions with parameters.
The boundary condition `func` is called with the signature

    `func(i, j, grid, clock, model_fields, parameters)`

where `i, j` are the indices along the boundary,
where `grid` is `model.grid`, `clock.time` is the current simulation time and
`clock.iteration` is the current model iteration, and
`model_fields` is a `NamedTuple` with `u, v, w`, the fields in `model.tracers`,
and the fields in `model.diffusivities`, each of which is an `OffsetArray`s (or `NamedTuple`s
of `OffsetArray`s depending on the turbulence closure) of field data.

*Note* that the index `end` does *not* access the final physical grid point of
a model field in any direction. The final grid point must be explictly specified, as
in `model_fields.u[i, j, grid.Nz]`*.

Example
=======

@inline linear_bottom_drag(i, j, grid, clock, model_fields, parameters) = 
    @inbounds - parameters.μ * model_fields.u[i, j, 1]

u_boundary_condition = ParameterizedBoundaryCondition(linear_bottom_drag, (μ=π,))
"""
struct ParameterizedDiscreteBoundaryFunction{F, P} <: Function
    func :: F
    parameters :: P
end

@inline (bc::ParameterizedDiscreteBoundaryFunction)(args...) = bc.func(args..., bc.parameters)
