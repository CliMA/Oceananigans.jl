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
τˣ = -1e-4
Jᵇ = 1e-7

u_top_bc = FluxBoundaryCondition(τˣ)
u_bcs = FieldBoundaryConditions(top=u_top_bc)

b_top_bc = FluxBoundaryCondition(Jᵇ)
b_bcs = FieldBoundaryConditions(top=b_top_bc)

coriolis = FPlane(; f)

const_stability_functions = ConstantStabilityFunctions()
k_epsilon = TKEDissipationVerticalDiffusivity()
k_epsilon_const_stability = TKEDissipationVerticalDiffusivity(stability_functions=const_stability_functions)
catke = CATKEVerticalDiffusivity()

bᵢ(z) = N² * z

bn = []
un = []
vn = []
en = []
ϵn = []

#stratification_numberᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)

for closure in (k_epsilon, catke) #, k_epsilon_const_stability)
    global model

    model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                        tracers = (:b, :e, :ϵ),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions=(u=u_bcs, b=b_bcs))

    set!(model, b=bᵢ)

    simulation = Simulation(model, Δt=60.0, stop_time=1day)

    local b, u, v, w, e, ϵ

    u, v, w = model.velocities
    e = model.tracers.e
    ϵ = model.tracers.ϵ
    b = model.tracers.b

    progress(sim) = @info @sprintf("Iter: % 4d, time: % 24s, max(e): %6.2e, max(ϵ): %6.2e",
                                   iteration(sim), prettytime(sim), maximum(e), maximum(ϵ))
    
    add_callback!(simulation, progress, IterationInterval(100))

    run!(simulation)

    push!(bn, deepcopy(interior(b, 1, 1, :)))
    push!(un, deepcopy(interior(u, 1, 1, :)))
    push!(vn, deepcopy(interior(v, 1, 1, :)))
    push!(en, deepcopy(interior(e, 1, 1, :)))
    push!(ϵn, deepcopy(interior(ϵ, 1, 1, :)))
end

z = znodes(model.tracers.e)

colors = [:black, :blue, :tomato]

fig = Figure(size=(800, 400))

axb = Axis(fig[1, 1], title="Velocities")
axu = Axis(fig[1, 2], title="Velocities")
axe = Axis(fig[1, 3], title="TKE")
axϵ = Axis(fig[1, 4], title="Epsilon")

lines!(axb, bn[1], z, color=colors[1])
lines!(axu, un[1], z, color=colors[1], label="u")
lines!(axu, vn[1], z, color=colors[1], linestyle=:dash, label="v")
lines!(axe, en[1], z, color=colors[1], label="k-ϵ")
lines!(axϵ, ϵn[1], z, color=colors[1])

lines!(axb, bn[2], z, color=colors[2])
lines!(axu, un[2], z, color=colors[2], label="u")
lines!(axu, vn[2], z, color=colors[2], linestyle=:dash, label="v")
lines!(axe, en[2], z, color=colors[2], label="CATKE")
lines!(axϵ, ϵn[2], z, color=colors[2])

#=
lines!(axb, bn[3], z, color=colors[3])
lines!(axu, un[3], z, color=colors[3], label="u")
lines!(axu, vn[3], z, color=colors[3], linestyle=:dash, label="v")
lines!(axe, en[3], z, color=colors[3], label="k-ϵ (constant stability functions)")
lines!(axϵ, ϵn[3], z, color=colors[3])
=#

Legend(fig[0, 1:4], axe, nbanks=3, framevisible=false, tellheight=true)

fig

