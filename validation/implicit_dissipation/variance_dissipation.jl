using Oceananigans
using Oceananigans.Models.VarianceDissipationComputations
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper
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

tracer_advection = WENO(order=9)
closure = nothing # ScalarDiffusivity(κ=1e-5)
velocities = PrescribedVelocityFields(u=1)
    
function run_simulation(ts, timestepper)   
    c⁻    = CenterField(grid)
    Δtc²  = CenterField(grid)     

    model = HydrostaticFreeSurfaceModel(; grid, 
                                        timestepper, 
                                        velocities, 
                                        tracer_advection, 
                                        closure, 
                                        tracers=:c,
                                        auxiliary_fields=(; Δtc², c⁻))
                                            
    set!(model, c=c₀)
    set!(model.auxiliary_fields.c⁻, c₀)

    Δt  = 0.3 * minimum_xspacing(grid)
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

    c    = FieldTimeSeries("one_d_simulation_$(ts).jld2", "c")
    Δtc² = FieldTimeSeries("one_d_simulation_$(ts).jld2", "Δtc²")
    Acx  = FieldTimeSeries("one_d_simulation_$(ts).jld2", "Acx")
    Dcx  = FieldTimeSeries("one_d_simulation_$(ts).jld2", "Dcx")

    Nt = length(c.times)

    ∫closs = abs.([sum(interior(Δtc²[i], :, 1, 1) .* grid.Δxᶜᵃᵃ) for i in 2:Nt-1])
    ∫A     = abs.([sum(interior(Acx[i] , :, 1, 1))               for i in 2:Nt-1])
    ∫D     = abs.([sum(interior(Dcx[i] , :, 1, 1))               for i in 2:Nt-1])
    ∫T     = ∫D .+ ∫A
    times  = c.times[2:end-1]

    return (; c, Δtc², Acx, Dcx, ∫closs, ∫A, ∫D, ∫T, times)
end

cases = Dict()

cases["AB2"]  = run_simulation(:AB2, :QuasiAdamsBashforth2)
cases["RK2"]  = run_simulation(:RK2, :SplitRungeKutta2)
cases["RK3"]  = run_simulation(:RK3, :SplitRungeKutta3)
cases["RK4"]  = run_simulation(:RK4, :SplitRungeKutta4)
cases["RK5"]  = run_simulation(:RK5, :SplitRungeKutta5)
cases["RK6"]  = run_simulation(:RK6, :SplitRungeKutta6)
cases["RK7"]  = run_simulation(:RK7, :SplitRungeKutta7)
cases["RK8"]  = run_simulation(:RK8, :SplitRungeKutta8)
cases["RK9"]  = run_simulation(:RK9, :SplitRungeKutta9)
cases["RK10"] = run_simulation(:RK10, :SplitRungeKutta10)
cases["RK20"] = run_simulation(:RK20, :SplitRungeKutta20)
cases["RK30"] = run_simulation(:RK30, :SplitRungeKutta30)
cases["RK40"] = run_simulation(:RK40, :SplitRungeKutta40)

fig = Figure()
ax  = Axis(fig[1, 1], title="Dissipation", xlabel="Time (s)", ylabel="Dissipation", yscale=log10)

for key in keys(cases)
    case = cases[key]
    scatter!(ax, case.times, case.∫closs, label="$(key) total variance loss")
    lines!(ax, case.times, case.∫T, label="$(key) total dissipation")
end
Legend(fig[1, 2], ax)