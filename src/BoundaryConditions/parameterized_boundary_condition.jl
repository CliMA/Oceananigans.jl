"""
    ParameterizedDiscreteBoundaryFunction(func, parameters)

A wrapper for boundary condition functions implemented as functions with parameters.
The boundary condition `func` is called with the signature

    `func(i, j, grid, clock, state, parameters)`

where `i, j` are the indices along the boundary, `grid` and `clock` are `model.grid` and
`model.clock`, and `state` is a `NamedTuple` with fields `velocities`, `tracers`,
and `diffusivities`, which are each `NamedTuple`s of `OffsetArray`s that reference
the data associated with their corresponding fields.

Example
=======

@inline linear_drag(i, j, grid, clock, state, parameters) = 
    @inbounds - parameters.μ * state.velocities.u[i, j, 1]

u_boundary_condition = ParameterizedBoundaryCondition(linear_drag, (μ=π,))
"""
struct ParameterizedDiscreteBoundaryFunction{F, P} <: Function
    func :: F
    parameters :: P
end

@inline (bc::ParameterizedDiscreteBoundaryFunction)(args...) = bc.func(args..., bc.parameters)
