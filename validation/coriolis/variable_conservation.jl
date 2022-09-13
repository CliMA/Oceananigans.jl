using Oceananigans
using Oceananigans.Coriolis

using Oceananigans.Units

using Oceananigans.Advection
using Oceananigans.Advection: EnergyConservingScheme, EnstrophyConservingScheme
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

grid = LatitudeLongitudeGrid(size = (10, 70, 1), 
                         latitude = (-70, 70), 
                        longitude = (-5, 5), 
                                z = (0, 1),
                         topology = (Periodic, Bounded, Bounded))

schemes = [WENO(), EnstrophyConservingScheme(), EnergyConservingScheme(), WENO()]

coriolis = HydrostaticSphericalCoriolis(scheme = schemes[1])

model = HydrostaticFreeSurfaceModel(; grid, coriolis, momentum_advection = nothing, tracers = (), buoyancy = nothing)

initial_u(x, y, z) = y < 45 ? (y > 0 ? 1.0 : (y > - 45 ?   1.0 : 0.0)) : 0.0
initial_v(x, y, z) = y < 45 ? (y > 0 ? 1.0 : (y > - 45 ?  -1.0 : 0.0)) : 0.0

set!(model, u = initial_u, v = initial_v)

using Oceananigans.Fields: @compute

u, v, w = model.velocities

using Oceananigans.AbstractOperations: volume

@compute ke = Field(u^2 + v^2)
ζ = VerticalVorticityField(model)

@compute ω = Field(ζ * ζ)
initial_k = sum(volume * ke)
initial_ω = sum(volume * ω)

for step in 1:10
    @info "step number $step"
    time_step!(model, 10minutes)
end

compute!(ke)
compute!(ω)

final_k = sum(volume * ke)
final_ω = sum(volume * ω)