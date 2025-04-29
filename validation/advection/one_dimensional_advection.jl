using Oceananigans
using Oceananigans.Diagnostics: VarianceDissipation

N = 200

# 1D grid constructions
grid = RectilinearGrid(size=N, x=(-1, 1), halo=7, topology = (Periodic, Flat, Flat))

# the initial condition
@inline G(x, β, z) = exp(-β*(x - z)^2)
@inline F(x, α, a) = √(max(1 - α^2*(x-a)^2, 0.0))

Z = -0.7
δ = 0.005
β = log(2)/(36*δ^2)
a = 0.5
α = 10

@inline function c₀(x)
    if x <= -0.6 && x >= -0.8
        return 1/6*(G(x, β, Z-δ) + 4*G(x, β, Z) + G(x, β, Z+δ))
    elseif x <= -0.2 && x >= -0.4
        return 1.0
    elseif x <= 0.2 && x >= 0.0
        return 1.0 - abs(10 * (x - 0.1))
    elseif x <= 0.6 && x >= 0.4
        return 1/6*(F(x, α, a-δ) + 4*F(x, α, a) + F(x, α, a+δ))
    else
        return 0.0
    end
end

Δt_max = 0.1 * minimum_xspacing(grid)
c_real = CenterField(grid)
set!(c_real, c₀)

# Change to test pure advection schemes
tracer_advection = WENO()
closure = nothing # ScalarDiffusivity(κ = 1e-5)
velocities = PrescribedVelocityFields(u=1)

model = HydrostaticFreeSurfaceModel(; grid, timestepper=:QuasiAdamsBashforth2, velocities, tracer_advection, closure, tracers=:c)
Oceananigans.BoundaryConditions.fill_halo_regions!(model.velocities.u)

c⁻    = CenterField(grid)
Δtc²  = CenterField(grid)

set!(model, c=c₀)
set!(c⁻, c₀)

sim = Simulation(model, Δt=Δt_max, stop_time=10)

ϵ = VarianceDissipation(model)
f = Oceananigans.Diagnostics.VarianceDissipationComputations.flatten_dissipation_fields(ϵ)
outputs = merge((; c = model.tracers.c, Δtc²), f)
sim.diagnostics[:variance_dissipation] = ϵ

sim.output_writers[:solution] = JLD2Writer(model, outputs;
                                           filename="one_d_simulation.jld2",
                                           schedule=TimeInterval(0.1),
                                           overwrite_existing=true)

function compute_tracer_dissipation(sim)
    c = sim.model.tracers.c
    parent(Δtc²) .= (parent(c).^2 .- parent(c⁻).^2) ./ sim.Δt
    parent(c⁻)   .= parent(c).^2 
end

sim.callbacks[:compute_tracer_dissipation] = Callback(compute_tracer_dissipation, IterationInterval(1))

run!(sim)

∫c1 = [sum(interior(c[i])) - sum(interior(c[1])) for i in 1:Nt]

c = FieldTimeSeries("one_d_simulation.jld2", "c")
Δtc² = FieldTimeSeries("one_d_simulation.jld2", "Δtc²")
Acx  = FieldTimeSeries("one_d_simulation.jld2", "Acx")
Dcx  = FieldTimeSeries("one_d_simulation.jld2", "Dcx")

using GLMakie

# iter = Observable(1)
Nt = length(c.times)

# fig = Figure()
# ax = Axis(fig[1, 1], title="c", xlabel="x", ylabel="t")
# ci = @lift(c[$iter])
# Δtc²i = @lift(Δtc²[$iter])
# Acxi  = @lift(Acx[$iter])
# Dcxi  = @lift(Dcx[$iter])
# lines!(ax, Δtc²i, color=:blue)
# lines!(ax, Acxi, color=:red)
# lines!(ax, Dcxi, color=:green)
# ylims!(ax, (-1, 1))

# GLMakie.record(fig, "one_d_simulation.mp4", 1:Nt) do i
#     iter[] = i
# end

int(c) = compute!(Field(Integral(c)))[1, 1, 1]

∫closs = [int(c[i]^2) - int(c[1]^2) for i in 1:Nt]
∫A = [sum(int(Acx[j]) for j in 1:i) for i in 1:Nt]
∫D = [sum(int(Dcx[j]) for j in 1:i) for i in 1:Nt]