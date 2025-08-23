using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner, compute_laplacian!
using Oceananigans.Grids: with_number_type, XYZRegularRG
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using Statistics
using CairoMakie
using Random

rng = Xoshiro(123)

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.55
    bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)

    set!(model, b=bᵢ)
end

function setup_grid(Nx, Ny, Nz, arch)
    zs = collect(range(0, 1, length=Nz+1))
    zs[2:end-1] .+= randn(length(zs[2:end-1])) * (1 / Nz) / 10

    grid = RectilinearGrid(arch, Float64,
                        size = (Nx, Ny, Nz), 
                        halo = (4, 4, 4),
                        x = (0, 1),
                        y = (0, 1),
                        z = (0, 1),
                        # z = zs,
                        topology = (Bounded, Bounded, Bounded))

    slope(x, y) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) + tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 20 + 
                  (5 + tanh(40*(y - 1/6)) + tanh(40*(y - 2/6)) + tanh(40*(y - 3/6)) + tanh(40*(y - 4/6)) + tanh(40*(y - 5/6))) / 20

    # grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
    grid = ImmersedBoundaryGrid(grid, PartialCellBottom(slope))
    return grid
end

function setup_model(grid, pressure_solver)
    model = NonhydrostaticModel(; grid, pressure_solver,
                                  advection = WENO(),
                                  coriolis = FPlane(f=0.1),
                                  tracers = :b,
                                  buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

@kernel function _divergence!(target_field, u, v, w, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds target_field[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_flow_divergence!(target_field, model)
    grid = model.grid
    u, v, w = model.velocities
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _divergence!, target_field, u, v, w, grid)
    return nothing
end


function setup_simulation(model)
    Δt = 1e-3
    simulation = Simulation(model; Δt = Δt, stop_time = 10, minimum_relative_step = 1e-10)
    conjure_time_step_wizard!(simulation, cfl=0.7, IterationInterval(1))
    
    wall_time = Ref(time_ns())

    d = Field{Center, Center, Center}(grid)

    function progress(sim)
        pressure_solver = sim.model.pressure_solver
    
        if pressure_solver isa ConjugateGradientPoissonSolver
            pressure_iters = iteration(pressure_solver)
        else
            pressure_iters = 0
        end

        msg = @sprintf("iter: %d, time: %s, Δt: %.4f, Poisson iters: %d",
                        iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)
    
        elapsed = 1e-9 * (time_ns() - wall_time[])
    
        compute_flow_divergence!(d, sim.model)
    
        msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max b: %6.3e, max d: %6.3e, max pressure: %6.3e, wall time: %s",
                        maximum(sim.model.velocities.u),
                        maximum(sim.model.velocities.v),
                        maximum(sim.model.velocities.w),
                        maximum(sim.model.tracers.b),
                        maximum(d),
                        maximum(sim.model.pressures.pNHS),
                        prettytime(elapsed))
    
        @info msg
        wall_time[] = time_ns()
    
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

    compute_flow_divergence!(d, model)

    B = Field(Integral(model.tracers.b))

    outputs = merge(model.velocities, model.tracers, (; p=model.pressures.pNHS, d, B))

    if grid.underlying_grid isa XYZRegularRG
        file_prefix = "uniform_"
    else
        file_prefix = "nonuniform_"
    end

    file_prefix *= "staircase_2D_convection"

    if preconditioner isa FFTBasedPoissonSolver
        file_prefix *= "_cgfft"
    elseif preconditioner isa FourierTridiagonalPoissonSolver
        file_prefix *= "_cgftri"
    else
        file_prefix *= "_cgnoprec"
    end

    if grid.immersed_boundary isa PartialCellBottom
        file_prefix *= "_partialcellbottom"
    else
        file_prefix *= "_gridfittedbottom"
    end

    filename = "./$(file_prefix)"
    simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                                filename = filename,
                                                schedule = TimeInterval(0.1),
                                                overwrite_existing = true)
    
    return simulation, file_prefix
end

arch = GPU()
Nx = Ny = Nz = 32
grid = setup_grid(Nx, Ny, Nz, arch)

@info "Create pressure solver"

# preconditioner = nonhydrostatic_pressure_solver(grid)
preconditioner = nonhydrostatic_pressure_solver(with_number_type(Float32, grid.underlying_grid))
# preconditioner = nothing

pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=10000; preconditioner)

model = setup_model(grid, pressure_solver)

simulation, filename = setup_simulation(model)

run!(simulation)

#%%
bt = FieldTimeSeries(filename, "b")
ut = FieldTimeSeries(filename, "u")
wt = FieldTimeSeries(filename, "w")
pt = FieldTimeSeries(filename, "p")
δt = FieldTimeSeries(filename, "d")
times = bt.times
Nt = length(times)

Bt = FieldTimeSeries(filename, "B")
#%%
yloc = Nz ÷ 2

fig = Figure(size=(1200, 1200))

n = Observable(1)

B₀ = sum(interior(bt[1], :, 1, :)) / (Nx * Nz)
btitlestr = @lift @sprintf("Buoyancy at t = %.2f", times[$n])
utitlestr = @lift @sprintf("Horizontal velocity at t = %.2f", times[$n])
wtitlestr = @lift @sprintf("Vertical velocity at t = %.2f", times[$n])

δlim = 1e-9

axb = Axis(fig[1, 1], title=btitlestr)
axu = Axis(fig[1, 2], title=utitlestr)
axw = Axis(fig[1, 3], title=wtitlestr)
axp = Axis(fig[2, 1], title="Pressure")
axd = Axis(fig[2, 2], title="Divergence, lim = $(δlim)")
axt = Axis(fig[3, 1:3], xlabel="Time", ylabel="Fractional remaining tracer")

bn = @lift interior(bt[$n], :, yloc, :)
un = @lift interior(ut[$n], :, yloc, :)
wn = @lift interior(wt[$n], :, yloc, :)
pn = @lift interior(pt[$n], :, yloc, :)
δn = @lift interior(δt[$n], :, yloc, :)

ulim = maximum(abs, ut) / 2
wlim = maximum(abs, wt) / 2
plim = maximum(abs, pt) / 2

heatmap!(axb, bn, colormap=:balance, colorrange=(-0.5, 0.5))
heatmap!(axu, un, colormap=:balance, colorrange=(-ulim, ulim))
heatmap!(axw, wn, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axp, pn, colormap=:balance, colorrange=(-plim, plim))
heatmap!(axd, δn, colormap=:balance, colorrange=(-δlim, δlim))

ΔB = Bt.data[1, 1, 1, :] .- Bt.data[1, 1, 1, 1]
t = @lift times[$n]
lines!(axt, times, ΔB)
vlines!(axt, t, color=:black)
# display(fig)

CairoMakie.record(fig, "./$(filename).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end
