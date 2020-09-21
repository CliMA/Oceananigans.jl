using Oceananigans.Fields: ComputedField

function PressureField(model, data=model.pressures.pHY′.data)
    p = ComputedField(model.pressures.pHY′ + model.pressures.pNHS, data=data)
    return p
end
