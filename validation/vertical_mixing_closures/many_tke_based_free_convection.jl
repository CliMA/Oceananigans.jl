pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity, TKESurfaceFlux
using Oceananigans.TurbulenceClosures: RiDependentDiffusivityScaling
using Oceananigans.Models.HydrostaticFreeSurfaceModels: EnsembleSize

Nz = 16
Ex, Ey = (1, 3)
sz = EnsembleSize(Nz=Nz, ensemble=(Ex, Ey))
ensemble_grid = RegularRectilinearGrid(size=sz, halo=EnsembleSize(Nz=1), z=(-64, 0), topology=(Flat, Flat, Bounded))
scm_grid = RegularRectilinearGrid(size=Nz, halo=1, z=(-64, 0), topology=(Flat, Flat, Bounded))

default_closure = TKEBasedVerticalDiffusivity()

closure_ensemble = [default_closure for i = 1:Ex, j = 1:Ey]

Qᵇ = 1e-8
Qᵇ_ensemble = [1e-8 for i = 1:Ex, j = 1:Ey]

closure_ensemble[1, 2] = TKEBasedVerticalDiffusivity(surface_model=TKESurfaceFlux(CᵂwΔ=3.0))
closure_ensemble[1, 3] = TKEBasedVerticalDiffusivity(surface_model=TKESurfaceFlux(CᵂwΔ=10.0))
                                      
Qᵘ = 0.0
Qᵛ = 0.0

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ_ensemble))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model_kwargs = (tracers = (:b, :e),
                buoyancy = BuoyancyTracer())

ensemble_model = HydrostaticFreeSurfaceModel(; grid = ensemble_grid, closure = closure_ensemble,       boundary_conditions = (; b=ensemble_b_bcs), model_kwargs...)
scm_model      = HydrostaticFreeSurfaceModel(; grid = scm_grid,      closure = closure_ensemble[1, 1], boundary_conditions = (; b=b_bcs),          model_kwargs...)

models = (ensemble_model, scm_model)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z

for model in models
    set!(model, b = bᵢ)
end

z = znodes(scm_model.tracers.b)

b1 = view(interior(scm_model.tracers.b), 1, 1, :)

b2 = view(interior(ensemble_model.tracers.b), 1, 1, :)
b3 = view(interior(ensemble_model.tracers.b), 1, 2, :)
b4 = view(interior(ensemble_model.tracers.b), 1, 3, :)

Kc1 = view(interior(scm_model.diffusivity_fields.Kᶜ), 1, 1, :)

Kc2 = view(interior(ensemble_model.diffusivity_fields.Kᶜ), 1, 1, :)
Kc3 = view(interior(ensemble_model.diffusivity_fields.Kᶜ), 1, 2, :)
Kc4 = view(interior(ensemble_model.diffusivity_fields.Kᶜ), 1, 3, :)

for model in models
    simulation = Simulation(model, Δt = 20.0, stop_iteration = 10)
    run!(simulation)
end

time = scm_model.clock.time

b_plot = plot(b1, z, linewidth = 2, label = @sprintf("scm t = %s", prettytime(time)), xlabel = "Buoyancy 1", ylabel = "z", legend=:bottomright)
plot!(b_plot, b2, z, linewidth = 2, linestyle=:dash, label = @sprintf("ensemble t = %s", prettytime(time)))
plot!(b_plot, b3, z, linewidth = 2, linestyle=:dash, label = @sprintf("ensemble t = %s", prettytime(time)))
plot!(b_plot, b4, z, linewidth = 2, linestyle=:dash, label = @sprintf("ensemble t = %s", prettytime(time)))

K_plot = plot(Kc1, z, linewidth = 2, linestyle=:solid, label = @sprintf("scm Kᶜ, t = %s", prettytime(time)), legend=:bottomright, xlabel="Diffusivities")
plot!(K_plot, Kc2, z, linewidth = 2, linestyle=:dash, label = @sprintf("ensemble Kᶜ, t = %s", prettytime(time)))
plot!(K_plot, Kc3, z, linewidth = 2, linestyle=:dash, label = @sprintf("ensemble Kᶜ, t = %s", prettytime(time)))
plot!(K_plot, Kc4, z, linewidth = 2, linestyle=:dash, label = @sprintf("ensemble Kᶜ, t = %s", prettytime(time)))

bK_plot = plot(b_plot, K_plot, layout=(1, 2), size=(1200, 600))

display(bK_plot)
