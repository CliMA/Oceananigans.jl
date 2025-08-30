using Oceananigans

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

Δt_max = 0.5 * minimum_xspacing(grid)
c_real = CenterField(grid)
set!(c_real, c₀)

# Change to test pure advection schemes
advection = WENO(order=7)

model1 = NonhydrostaticModel(; grid, timestepper=:RungeKutta3, advection, tracers=:c)
set!(model1, c=c₀, u=1)
sim1 = Simulation(model1, Δt=Δt_max, stop_time=10)

model2 = HydrostaticFreeSurfaceModel(; grid, velocities=PrescribedVelocityFields(u=1), timestepper=:SplitRungeKutta3, tracer_advection=advection, tracers=:c)
set!(model2, c=c₀)
sim2 = Simulation(model2, Δt=Δt_max, stop_time=10)

sim1.output_writers[:solution] = JLD2Writer(model1, (; c = model1.tracers.c);
                                            filename="one_d_simulation_NH.jld2",
                                            schedule=IterationInterval(10),
                                            overwrite_existing=true)

sim2.output_writers[:solution] = JLD2Writer(model2, (; c = model2.tracers.c);
                                            filename="one_d_simulation_HF.jld2",
                                            schedule=IterationInterval(10),
                                            overwrite_existing=true)

run!(sim1)
run!(sim2)

c1 = FieldTimeSeries("one_d_simulation_NH.jld2", "c")
c2 = FieldTimeSeries("one_d_simulation_HF.jld2", "c")

using GLMakie

iter = Observable(1)
Nt = length(c1.times)

fig = Figure()
ax = Axis(fig[1, 1], title="c", xlabel="x", ylabel="t")
c1i = @lift(interior(c1[$iter], :, 1, 1))
c2i = @lift(interior(c2[$iter], :, 1, 1))
lines!(ax, c1i, color=:blue, label = "RK3  (NonhydrostaticModel)")
lines!(ax, c2i, color=:red,  label = "SRK3 (HydrostaticFreeSurfaceModel)")
Legend(fig[0, 1], ax)

GLMakie.record(fig, "one_d_simulation.mp4", 1:Nt) do i
    iter[] = i
end