pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, SurfaceTKEFlux, MixingLength
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize

Nz = 64
Ex, Ey = (1, 3)
sz = ColumnEnsembleSize(Nz=Nz, ensemble=(Ex, Ey))

ensemble_grid = RectilinearGrid(size = sz,
                                       halo = ColumnEnsembleSize(Nz=1),
                                       z = (-128, 0),
                                       topology = (Flat, Flat, Bounded))

default_closure = CATKEVerticalDiffusivity()
closure_ensemble = [default_closure for i = 1:Ex, j = 1:Ey]

Qᵇ = 1e-7
Qᵇ_ensemble = [Qᵇ for i = 1:Ex, j = 1:Ey]



closure_ensemble[1, 1] = CATKEVerticalDiffusivity(mixing_length = MixingLength(Cᴷc⁻=1.0, Cᴸᵇ=0.0))
closure_ensemble[1, 2] = CATKEVerticalDiffusivity(mixing_length = MixingLength(Cᴷc⁻=1.0, Cᴸᵇ=2.0))
closure_ensemble[1, 3] = CATKEVerticalDiffusivity(mixing_length = MixingLength(Cᴷc⁻=1.0, Cᴸᵇ=4.0))
                                      
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ_ensemble))

model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                    closure = closure_ensemble, 
                                    boundary_conditions = (; b=b_bcs),
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer())

N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

simulation = Simulation(model, Δt = 1minute/2, stop_time = 0.0)

b = model.tracers.b
bz = ComputedField(∂z(b))

function column_bz(j)
    compute!(bz)
    return view(interior(bz), 1, j, :)
end

Nt = 10
hs = zeros(Ey, Nt)
times = zeros(Nt)
for n = 1:Nt
    simulation.stop_time += 4hour
    run!(simulation)

    h = [-z[argmax(column_bz(j))] for j = 1:3]

    hs[:, n] .= h
    times[n] = model.clock.time
end

b1 = view(interior(b), 1, 1, :)
b2 = view(interior(b), 1, 2, :)
b3 = view(interior(b), 1, 3, :)

compute!(bz)
zbz = znodes(bz)

bz1 = view(interior(bz), 1, 1, :)
bz2 = view(interior(bz), 1, 2, :)
bz3 = view(interior(bz), 1, 3, :)

t = model.clock.time
b_plot = plot(b1, z, linewidth = 2, label = @sprintf("b1 t = %s", prettytime(t)), xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
plot!(b_plot, b2, z, linewidth = 2, linestyle=:dash, label = @sprintf("b2 t = %s", prettytime(t)))
plot!(b_plot, b3, z, linewidth = 2, linestyle=:dot, label = @sprintf("b3 t = %s", prettytime(t)))

bz_plot = plot(bz1, zbz, linewidth = 2, label = @sprintf("bz1 t = %s", prettytime(t)), xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
plot!(bz_plot, bz2, zbz, linewidth = 2, linestyle=:dash, label = @sprintf("bz2 t = %s", prettytime(t)))
plot!(bz_plot, bz3, zbz, linewidth = 2, linestyle=:dot, label = @sprintf("bz3 t = %s", prettytime(t)))

bbz_plot = plot(b_plot, bz_plot, layout=(1, 2), size=(1200, 600))

display(bbz_plot)
