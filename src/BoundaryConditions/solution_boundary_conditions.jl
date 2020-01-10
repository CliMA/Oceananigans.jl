"""
    SolutionBoundaryConditions(tracers, proposal_bcs)

Construct a `NamedTuple` of `FieldBoundaryConditions` for a model with
fields `u`, `v`, `w`, and `tracers` from the proposal boundary conditions
`proposal_bcs`, which must contain the boundary conditions on `u`, `v`, and `w`
and may contain some or all of the boundary conditions on `tracers`.
"""
SolutionBoundaryConditions(tracers, proposal_bcs) =
    with_tracers(tracers, proposal_bcs, default_tracer_bcs, with_velocities=true)

"""
    HorizontallyPeriodicSolutionBCs(u=HorizontallyPeriodicBCs(), ...)

Construct `SolutionBoundaryConditions` for a horizontally-periodic model
configuration with solution fields `u`, `v`, `w`, `T`, and `S` specified by keyword arguments.

By default `HorizontallyPeriodicBCs` are applied to `u`, `v`, `T`, and `S`
and `HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())` is applied to `w`.

Use `HorizontallyPeriodicBCs` when constructing non-default boundary conditions for `u`, `v`, `w`, `T`, `S`.
"""
function HorizontallyPeriodicSolutionBCs(;
    u = HorizontallyPeriodicBCs(),
    v = HorizontallyPeriodicBCs(),
    w = HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    tracerbcs...)

    return merge((u=u, v=v, w=w), tracerbcs)
end

"""
    ChannelSolutionBCs(u=ChannelBCs(), ...)

Construct `SolutionBoundaryConditions` for a reentrant channel model
configuration with solution fields `u`, `v`, `w`, `T`, and `S` specified by keyword arguments.

By default `ChannelBCs` are applied to `u`, `v`, `T`, and `S`
and `ChannelBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())` is applied to `w`.

Use `ChannelBCs` when constructing non-default boundary conditions for `u`, `v`, `w`, `T`, `S`.
"""
function ChannelSolutionBCs(;
    u = ChannelBCs(),
    v = ChannelBCs(north=NoPenetrationBC(), south=NoPenetrationBC()),
    w = ChannelBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    tracerbcs...)

    return merge((u=u, v=v, w=w), tracerbcs)
end
