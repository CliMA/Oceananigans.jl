using Oceananigans
using GLMakie
using Printf
using Oceananigans.Models.NonhydrostaticModels: ImmersedPoissonSolver, DiagonallyDominantThreeDimensionalPreconditioner

#####
##### Model setup
#####

Nx = 64
Ny = 1
Nz = 64

grid = RectilinearGrid(size = (Nx, Ny, Nz), 
                       halo = (4, 4, 4),
                       x = (0, 1),
                       y = (0, 1),
                       z = (0, 1),
                       topology = (Bounded, Periodic, Bounded))

# function slope(x, y, n_steps, step_width, Lz, Lx)
#     tanh_centers = range(0, stop=Lx, length=n_steps+1)[2:end-1]

#     return sum([(tanh(1 / step_width * (x - center))) / (2*n_steps) for center in tanh_centers]) + 1 / 2
# end

slope(x, y) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) + tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 10

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
                       
@info "Created $grid"
model = NonhydrostaticModel(; grid,
                            pressure_solver = ImmersedPoissonSolver(grid, 
                                                                    solver_method = :HeptadiagonalIterativeSolver,
                                                                    reltol = 1e-8,
                                                                    preconditioner = "FFT",
                                                                    verbose = true),
                            advection = WENO(),
                            coriolis = FPlane(f=0.1),
                            tracers = (:b, :c),
                            buoyancy = BuoyancyTracer())

@info "Created $model"
@info "with pressure solver $(model.pressure_solver)"

# Cold blob
h = 0.03
x₀ = 0.75
z₀ = 0.9
bᵢ(x, y, z) = -2 * exp(-((x - x₀)^2 + (z - z₀)^2) / 2h^2)
set!(model, b=bᵢ, c=1)

#####
##### Simulation
#####

simulation = Simulation(model, Δt=2e-2, stop_iteration=1000) #stop_time = 10)

wall_time = Ref(time_ns())

b, c = model.tracers
u, v, w = model.velocities
B = Field(Integral(b))
C = Field(Integral(c))
compute!(B)
compute!(C)
B₀ = B[1, 1, 1]
C₀ = C[1, 1, 1]
δ = ∂x(u) + ∂y(v) + ∂z(w)

function progress(sim)
    elapsed = time_ns() - wall_time[]

    compute!(B)
    compute!(C)
    Bₙ = B[1, 1, 1]
    Cₙ = C[1, 1, 1]

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, ΔB: %.3f %%, ΔC: %.3f %%, max|δ|: %.2e",
                   iteration(sim), prettytime(sim), prettytime(1e-9 * elapsed),
                   100 * (Bₙ - B₀) / B₀,
                   100 * (Cₙ - C₀) / C₀,
                   maximum(abs, δ))

    pressure_solver = sim.model.pressure_solver
    # if sim.model.pressure_solver isa ImmersedPoissonSolver
    #     solver_iterations = pressure_solver.pcg_solver.iteration 
    #     msg *= string(", solver iterations: ", solver_iterations)
    # end

    @info msg

    wall_time[] = time_ns()

    return nothing
end
                   
simulation.callbacks[:p] = Callback(progress, IterationInterval(1))

solver_type = model.pressure_solver isa ImmersedPoissonSolver ? "ImmersedPoissonSolver" : "FFTBasedPoissonSolver"
prefix = "staircase_convection_" * solver_type

outputs = merge(model.velocities, model.tracers, (; p=model.pressures.pNHS, δ))

simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs;
                                                    filename = prefix * "_fieldss",
                                                   # schedule = TimeInterval(0.1),
                                                    schedule = IterationInterval(1),
                                                    overwrite_existing = true)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B, C);
                                                          filename = prefix * "_time_seriess",
                                                          schedule = IterationInterval(1),
                                                          overwrite_existing = true)

run!(simulation)

#####
##### Visualize
#####

filename = prefix * "_fieldss.jld2"
bt = FieldTimeSeries(filename, "b")
ct = FieldTimeSeries(filename, "c")
wt = FieldTimeSeries(filename, "w")
pt = FieldTimeSeries(filename, "p")
δt = FieldTimeSeries(filename, "δ")
times = bt.times
Nt = length(times)

time_series_filename = prefix * "_time_seriess.jld2"
Bt = FieldTimeSeries(time_series_filename, "B")
Ct = FieldTimeSeries(time_series_filename, "C")

fig = Figure(resolution=(1200, 1200))

slider = Slider(fig[0, 1:3], range=1:Nt, startvalue=1)
n = slider.value

B₀ = sum(interior(bt[1], :, 1, :)) / (Nx * Nz)
btitlestr = @lift @sprintf("Buoyancy at t = %.2f", times[$n])
wtitlestr = @lift @sprintf("Vertical velocity at t = %.2f", times[$n])

axb = Axis(fig[1, 1], title=btitlestr)
axw = Axis(fig[1, 2], title=wtitlestr)
axc = Axis(fig[1, 3], title="Passive tracer anomaly")
axp = Axis(fig[2, 1], title="Pressure")
axd = Axis(fig[2, 2], title="Divergence")
axt = Axis(fig[3, 1:3], xlabel="Time", ylabel="Fractional remaining tracer")

bn = @lift interior(bt[$n], :, 1, :)
c′n = @lift interior(ct[$n], :, 1, :) .- interior(ct[1], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)
pn = @lift interior(pt[$n], :, 1, :)
δn = @lift interior(δt[$n], :, 1, :)

wlim = maximum(abs, wt) / 2
plim = maximum(abs, pt) / 2
clim = 1e-1
δlim = 1e-8

heatmap!(axb, bn, colormap=:balance, colorrange=(-0.1, 0.1))
heatmap!(axw, wn, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axp, pn, colormap=:balance, colorrange=(-plim, plim))
heatmap!(axd, δn, colormap=:balance, colorrange=(-δlim, δlim))
heatmap!(axc, c′n, colormap=:balance, colorrange=(-clim, clim))

ΔB = Bt.data[1, 1, 1, :] .- Bt.data[1, 1, 1, 1]
C₀ = Ct.data[1, 1, 1, 1]
ΔC = (Ct.data[1, 1, 1, :] .- C₀) / C₀
t = @lift times[$n]
lines!(axt, Ct.times, ΔC)
vlines!(axt, t)

display(fig)

# record(fig, prefix * "_slope.mp4", 1:Nt, framerate=30) do nn
#     @info string("Plotting frame ", nn, " of ", Nt)
#     n[] = nn
# end