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
advection = WENO(order=5)

model = NonhydrostaticModel(; grid, timestepper=:RungeKutta3, advection, tracers=:c)

set!(model, c=c₀, u=1)
sim = Simulation(model, Δt=Δt_max, stop_time=10)

sim.output_writers[:solution] = JLD2Writer(model, (; c = model.tracers.c);
                                           filename="one_d_simulation.jld2",
                                           schedule=TimeInterval(0.1),
                                           overwrite_existing=true)

run!(sim)

c = FieldTimeSeries("one_d_simulation.jld2", "c")

using GLMakie

iter = Observable(1)
Nt = length(c.times)

fig = Figure()
ax = Axis(fig[1, 1], title="c", xlabel="x", ylabel="t")
ci = @lift(c[$iter])
lines!(ax, ci, color=:blue)

GLMakie.record(fig, "one_d_simulation.mp4", 1:Nt) do i
    iter[] = i
end