using Oceananigans
using Oceananigans.Models.VarianceDissipationComputations
using KernelAbstractions: @kernel, @index
using GLMakie

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

tracer_advection = WENO(order=5)
closure = ScalarDiffusivity(κ=1e-5)
velocities = PrescribedVelocityFields(u=1)

c⁻    = CenterField(grid)
Δtc²  = CenterField(grid)

for (ts, timestepper) in zip((:AB2, :RK3), (:QuasiAdamsBashforth2, :SplitRungeKutta3))
    
    model = HydrostaticFreeSurfaceModel(; grid, 
                                        timestepper, 
                                        velocities, 
                                        tracer_advection, 
                                        closure, 
                                        tracers=:c,
                                        auxiliary_fields=(; Δtc², c⁻))
                                            
    set!(model, c=c₀)
    set!(model.auxiliary_fields.c⁻, c₀)

    if ts == :AB2
       Δt = 0.2 * minimum_xspacing(grid)
    else
       Δt = 0.2 * minimum_xspacing(grid)
    end

    sim = Simulation(model; Δt, stop_time=10)

    ϵ = VarianceDissipation(:c, grid)
    f = Oceananigans.Models.VarianceDissipationComputations.flatten_dissipation_fields(ϵ)

    outputs = merge((; c = model.tracers.c, Δtc² = model.auxiliary_fields.Δtc²), f)
    add_callback!(sim, ϵ, IterationInterval(1))

    sim.output_writers[:solution] = JLD2Writer(model, outputs;
                                            filename="one_d_simulation_$(ts).jld2",
                                            schedule=IterationInterval(100),
                                            overwrite_existing=true)

    sim.callbacks[:compute_tracer_dissipation] = Callback(compute_tracer_dissipation!, IterationInterval(1))
    
    run!(sim)
end

a_c    = FieldTimeSeries("one_d_simulation_AB2.jld2", "c")
a_Δtc² = FieldTimeSeries("one_d_simulation_AB2.jld2", "Δtc²")
a_Acx  = FieldTimeSeries("one_d_simulation_AB2.jld2", "Acx")
a_Dcx  = FieldTimeSeries("one_d_simulation_AB2.jld2", "Dcx")

r_c    = FieldTimeSeries("one_d_simulation_RK3.jld2", "c")
r_Δtc² = FieldTimeSeries("one_d_simulation_RK3.jld2", "Δtc²")
r_Acx  = FieldTimeSeries("one_d_simulation_RK3.jld2", "Acx")
r_Dcx  = FieldTimeSeries("one_d_simulation_RK3.jld2", "Dcx")

Nta = length(a_c.times)
Ntr = length(r_c.times)

a_∫closs = abs.([sum(interior(a_Δtc²[i], :, 1, 1) .* grid.Δxᶜᵃᵃ) for i in 2:Nta-1])
a_∫A     = abs.([sum(interior(a_Acx[i] , :, 1, 1))               for i in 2:Nta-1])
a_∫D     = abs.([sum(interior(a_Dcx[i] , :, 1, 1))               for i in 2:Nta-1])
a_∫T     = a_∫D .+ a_∫A

r_∫closs = abs.([sum(interior(r_Δtc²[i], :, 1, 1) .* grid.Δxᶜᵃᵃ) for i in 2:Ntr-1])
r_∫A     = abs.([sum(interior(r_Acx[i] , :, 1, 1))               for i in 2:Ntr-1])
r_∫D     = abs.([sum(interior(r_Dcx[i] , :, 1, 1))               for i in 2:Ntr-1])
r_∫T     = r_∫D .+ r_∫A

atimes = a_c.times[2:end-1]
rtimes = r_c.times[2:end-1]

fig = Figure()
ax  = Axis(fig[1, 1], title="Dissipation", xlabel="Time (s)", ylabel="Dissipation", yscale=log10)

scatter!(ax, atimes, a_∫closs, label="AB2 total variance loss", color=:blue)
lines!(ax, atimes, a_∫A, label="AB2 advection dissipation", color=:red)
lines!(ax, atimes, a_∫D, label="AB2 diffusive dissipation", color=:green)
lines!(ax, atimes, a_∫T, label="AB2 total dissipation", color=:purple)

scatter!(ax, rtimes, r_∫closs, label="RK3 total variance loss", color=:blue, marker=:diamond)
lines!(ax, rtimes, r_∫A, label="RK3 advection dissipation", color=:red, linestyle=:dash)
lines!(ax, rtimes, r_∫D, label="RK3 diffusive dissipation", color=:green, linestyle=:dash)
lines!(ax, rtimes, r_∫T, label="RK3 total dissipation", color=:purple, linestyle=:dash)
Legend(fig[1, 2], ax)