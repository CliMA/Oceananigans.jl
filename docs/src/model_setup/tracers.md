# Tracers

The tracers to be advected around can be specified via a list of symbols. By default the model evolves temperature and
salinity
```
tracers = (:T, :S)
```
but any number of arbitrary tracers can be appended to this list. For example, to evolve quantities $C_1$, CO₂, and
nitrogen as passive tracers you could set them up as
```
tracers = (:T, :S, :C₁, :CO₂, :nitrogen)
```

!!! info "Active vs. passive tracers"
    An active tracer typically denotes a tracer quantity that affects the fluid dynamics through buoyancy. In the ocean
    temperature and salinity are active tracers. Passive tracers, on the other hand, typically do not affect the fluid
    dynamics are are _passively_ advected around by the flow field.
