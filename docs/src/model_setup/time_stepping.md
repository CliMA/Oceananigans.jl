## Time stepping
Once you're ready to time step the model simply call
```
time_step!(model; Δt=10)
```
to take a single time step with step size 10. To take multiple time steps also pass an `Nt` keyword argument like
```
time_step!(model; Δt=10, Nt=50)
```

By default, `time_step!` uses a first-order forward Euler time step to take the first time step then uses a second-order
Adams-Bashforth method for the remaining time steps (which required knowledge of the previous time step). If you are
resuming time-stepping then you should not use a forward Euler initialization time step. This can be done via
```
time_step!(model; Δt=10)
time_step!(model; Δt=10, Nt=50, init_with_euler=false)
```

### Adaptive time stepping
Adaptive time stepping can be acomplished using the [`TimeStepWizard`](@ref). It can be used to compute time steps based
on capping the CFL number at some value. You must remember to update the time step every so often. For example, to cap
the CFL number at 0.3 and update the time step every 50 time steps:
```
wizard = TimeStepWizard(cfl=0.3, Δt=1.0, max_change=1.2, max_Δt=30.0)

while model.clock.time < end_time
    time_step!(model; Δt=wizard.Δt, Nt=50)
    update_Δt!(wizard, model)
end
```
See [`TimeStepWizard`](@ref) for documentation of other features and options.

!!! warn "Maximum CFL with second-order Adams-Bashforth time stepping"
    For stable time-stepping it is recommended to cap the CFL at 0.3 or smaller, although capping it at 0.5 works well
    for certain simulations. For some simulations, it may be neccessary to cap the CFL number at 0.1 or lower.

!!! warn "Adaptive time stepping with second-order Adams-Bashforth time stepping"
    You should use an initializer forward Euler time step whenever changing the time step (i.e. `init_with_euler=true`
    which is the default value).
