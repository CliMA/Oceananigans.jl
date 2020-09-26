using Oceananigans.Fields: ComputedField

PressureField(model, data=model.pressures.pHY′.data, recompute_safely=false) =
    ComputedField(model.pressures.pHY′ + model.pressures.pNHS, data=data,
                      recompute_safely=recompute_safely)
