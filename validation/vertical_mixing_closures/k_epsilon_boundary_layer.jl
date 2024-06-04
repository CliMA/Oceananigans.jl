using Oceananigans
using Oceananigans.Units

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    TKEDissipationVerticalDiffusivity,
    CATKEVerticalDiffusivity,
    VariableStabilityFunctions,
    minimum_stratification_number,
    maximum_shear_number,
    ConstantStabilityFunctions,
    stratification_numberᶜᶜᶠ,
    shear_numberᶜᶜᶠ

using GLMakie
using Printf

grid = RectilinearGrid(size=128, z=(-128, 0), topology=(Flat, Flat, Bounded))

f = 0
N² = 1e-5
τˣ = 0 #-1e-4
Jᵇ = 1e-7

u_top_bc = FluxBoundaryCondition(τˣ)
u_bcs = FieldBoundaryConditions(top=u_top_bc)

b_top_bc = FluxBoundaryCondition(Jᵇ)
b_bcs = FieldBoundaryConditions(top=b_top_bc)

coriolis = FPlane(; f)

closure = TKEDissipationVerticalDiffusivity()
#closure = CATKEVerticalDiffusivity()

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                    tracers = (:b, :e, :ϵ),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions=(u=u_bcs, b=b_bcs))

bᵢ(z) = N² * z
set!(model, b=bᵢ)

simulation = Simulation(model, Δt=60.0, stop_time=1day)

u, v, w = model.velocities
e = model.tracers.e
ϵ = model.tracers.ϵ
b = model.tracers.b

tracers = model.tracers
buoyancy = model.buoyancy
velocities = model.velocities 

κc = model.diffusivity_fields.κc

progress(sim) = @info @sprintf("Iter: % 4d, time: % 24s, max(e): %6.2e, extrema(ϵ): (%6.2e, %6.2e)",
                               iteration(sim), prettytime(sim), maximum(e), minimum(ϵ), maximum(ϵ))

add_callback!(simulation, progress, IterationInterval(100))

run!(simulation)

κcn = interior(κc, 1, 1, :)
bn = interior(b, 1, 1, :)
un = interior(u, 1, 1, :)
vn = interior(v, 1, 1, :)
en = interior(e, 1, 1, :)
ϵn = interior(ϵ, 1, 1, :)

zc = znodes(model.tracers.e)
zf = znodes(κc)

fig = Figure(size=(800, 400))

axb = Axis(fig[1, 1], title="Buoyancy")
axu = Axis(fig[1, 2], title="Velocities")
axe = Axis(fig[1, 3], title="TKE")
axϵ = Axis(fig[1, 4], title="Epsilon")
axκ = Axis(fig[1, 5], title="Diffusivity")
axα = Axis(fig[1, 6], title="αᴺ, αᴹ")

lines!(axb, bn, zc)
lines!(axu, un, zc, label="u")
lines!(axu, vn, zc, linestyle=:dash, label="v")
lines!(axe, en, zc, label="k-ϵ")
lines!(axϵ, ϵn, zc)
lines!(axκ, κcn, zf)

αᴺ_op = KernelFunctionOperation{Center, Center, Face}(stratification_numberᶜᶜᶠ, grid, closure, tracers, buoyancy)
αᴹ_op = KernelFunctionOperation{Center, Center, Face}(shear_numberᶜᶜᶠ, grid, closure, velocities, tracers, buoyancy)
αᴺ = Field(αᴺ_op)
αᴹ = Field(αᴹ_op)
compute!(αᴺ)
compute!(αᴹ)
αᴹn = interior(αᴹ, 1, 1, :)
αᴺn = interior(αᴺ, 1, 1, :)
lines!(axα, αᴺn, zf, label="αᴺ")
lines!(axα, αᴹn, zf, label="αᴹ")
axislegend(axα)

fig

