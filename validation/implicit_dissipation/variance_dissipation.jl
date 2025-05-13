using Oceananigans
using Oceananigans.Diagnostics: VarianceDissipation
using KernelAbstractions: @kernel, @index

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

Δt_max = 0.2 * minimum_xspacing(grid)

# Change to test pure advection schemes
tracer_advection = WENO(order = 9)
closure = ScalarDiffusivity(κ=1e-5)
velocities = PrescribedVelocityFields(u=1)

c⁻    = CenterField(grid)
Δtc²  = CenterField(grid)

model = HydrostaticFreeSurfaceModel(; grid, 
                                      timestepper=:QuasiAdamsBashforth2, 
                                      velocities, 
                                      tracer_advection, 
                                      closure, 
                                      tracers=:c,
                                      auxiliary_fields=(; Δtc², c⁻))

set!(model, c=c₀)
set!(model.auxiliary_fields.c⁻, c₀)

sim = Simulation(model, Δt=Δt_max, stop_time=10)

ϵ = VarianceDissipation(model)
f = Oceananigans.Diagnostics.VarianceDissipationComputations.flatten_dissipation_fields(ϵ)
outputs = merge((; c = model.tracers.c, Δtc² = model.auxiliary_fields.Δtc²), f)
sim.diagnostics[:variance_dissipation] = ϵ

sim.output_writers[:solution] = JLD2Writer(model, outputs;
                                           filename="one_d_simulation.jld2",
                                           schedule=IterationInterval(100),
                                           overwrite_existing=true)

@kernel function _compute_dissipation!(Δtc², c⁻, c, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Δtc²[i, j, k] = (c[i, j, k]^2 - c⁻[i, j, k]^2) / Δt
        c⁻[i, j, k]   = c[i, j, k]
    end
end

function compute_tracer_dissipation!(sim)
    c    = sim.model.tracers.c
    c⁻   = sim.model.auxiliary_fields.c⁻
    Δtc² = sim.model.auxiliary_fields.Δtc²
    Oceananigans.Utils.launch!(CPU(), sim.model.grid, :xyz, 
                               _compute_dissipation!, 
                               Δtc², c⁻, c, sim.Δt)

    return nothing
end

sim.callbacks[:compute_tracer_dissipation] = Callback(compute_tracer_dissipation!, IterationInterval(1))

run!(sim)

c    = FieldTimeSeries("one_d_simulation.jld2", "c")
Δtc² = FieldTimeSeries("one_d_simulation.jld2", "Δtc²")
Acx  = FieldTimeSeries("one_d_simulation.jld2", "Acx")
Dcx  = FieldTimeSeries("one_d_simulation.jld2", "Dcx")

Nt = length(c.times)

∫closs = [sum(interior(Δtc²[i], :, 1, 1))  for i in 1:Nt]
∫A     = [sum(interior(Acx[i] , :, 1, 1))  for i in 1:Nt]
∫D     = [sum(interior(Dcx[i] , :, 1, 1))  for i in 1:Nt] 

fig = Figure()
ax  = Axis(fig[1, 1], title="Dissipation", xlabel="Time (s)", ylabel="Dissipation")
scatter!(ax, c.times, ∫closs .* grid.Δxᶜᵃᵃ, label="total variance loss", color=:blue)
lines!(ax, c.times, ∫A, label="advection dissipation", color=:red)
lines!(ax, c.times, ∫D, label="diffusive dissipation", color=:green)
lines!(ax, c.times, ∫D .+ ∫A, label="total dissipation", color=:purple)