# Checkpointing

A checkpointer can be used to serialize the entire model state to a file from which the model can be restored at any
time. This is useful if you'd like to periodically checkpoint when running long simulations in case of crashes or
cluster time limits, but also if you'd like to restore from a checkpoint and try out multiple scenarios.

For example, to periodically checkpoint the model state to disk every 1,000,000 seconds of simulation time to files of
the form `model_checkpoint_iteration12500.jld2` where `12500` is the iteration number (automatically filled in)

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> using Oceananigans, Oceananigans.Units

julia> model = IncompressibleModel(grid=RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)));

julia> checkpointer = Checkpointer(model, schedule=TimeInterval(5years), prefix="model_checkpoint")
Checkpointer{TimeInterval, Vector{Symbol}}(TimeInterval(1.5768e8, 0.0), ".", "model_checkpoint", [:architecture, :grid, :clock, :coriolis, :buoyancy, :closure, :velocities, :tracers, :timestepper, :particles], false, false, false)
```

The default options should provide checkpoint files that are easy to restore from in most cases. For more advanced
options and features, see [`Checkpointer`](@ref).

## Restoring from a checkpoint file

To restore the model from a checkpoint file, for example `model_checkpoint_12345.jld2`, simply call

```julia
model = restore_from_checkpoint("model_checkpoint_12345.jld2")
```

which will create a new model object that is identical to the one that was serialized to disk. You can continue time
stepping after restoring from a checkpoint.

You can pass additional parameters to the `Model` constructor. See [`restore_from_checkpoint`](@ref) for more
information.

## Restoring with functions

JLD2 cannot serialize functions to disk. so if you used forcing functions, boundary conditions containing functions, or
the model included references to functions then they will not be serialized to the checkpoint file. When restoring from
a checkpoint file, any model property that contained functions must be manually restored via keyword arguments to
[`restore_from_checkpoint`](@ref).
