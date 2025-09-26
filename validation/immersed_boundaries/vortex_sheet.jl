using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators
using Oceananigans.Grids: with_number_type
using Printf
using Statistics
using CUDA

const Nx = 400
const Nz = 200
const Lz = 4.0

const Δx = Lz / Nz
const Lx = Nx * Δx

const Rₑ = 30.0

const ν = 1.0
const L₀ = 1.0
const U₀ = Rₑ * ν / L₀

const Δt = 0.4 * Δx^2 / ν

const r₀ = L₀ / 2
const x₀ = 2.0 * L₀
const z₀ = 0.0

const γ = 1.0

initial_u(x, z) = 1e-7 * rand() + U₀
c_tendency(x, z, t, c) = - γ / Δt * (c - sin(40π * z / Lz)) * exp(-γ * x/Δx)
u_tendency(x, z, t, u) = - γ / Δt * (u - U₀) * exp(-γ * x/Δx)

function is_obstacle(x, z)
    return (x - x₀)^2 + (z - z₀)^2 < r₀^2
end

function setup_grid()
    zs = collect(range(-Lz / 2, Lz / 2, length=Nz+1))
    zs[2:end-1] .+= randn(length(zs[2:end-1])) * (1 / Nz) / 10

    underlying_grid = RectilinearGrid(
        GPU(),
        size = (Nx, Nz),
        x = (0.0, Lx),
        z = (- Lz / 2, Lz / 2),
        # z = zs,               # if you want to use a non-uniform grid
        topology = (Periodic, Flat, Bounded),
        halo = (4, 4),
    )

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(is_obstacle))
    return grid
end

closure = ScalarDiffusivity(ν = ν)

c_forcing = Forcing(c_tendency, field_dependencies = (:c, ))
u_forcing = Forcing(u_tendency, field_dependencies = (:u, ))

grid = setup_grid()
preconditioner = nonhydrostatic_pressure_solver(with_number_type(Float32, grid.underlying_grid))
# preconditioner = DiagonallyDominantPreconditioner()
# preconditioner = nothing

pressure_solver = ConjugateGradientPoissonSolver(
    grid, maxiter=1000, preconditioner=preconditioner)

model = NonhydrostaticModel(;
    grid,
    advection = WENO(),
    tracers = (:c, ),
    forcing = (; c = c_forcing, u = u_forcing),
    pressure_solver = pressure_solver,
)

set!(model, u = initial_u)

# simulation = Oceananigans.Simulation(model; Δt = Δt, stop_iteration = 10, minimum_relative_step = 1e-10)
simulation = Oceananigans.Simulation(model; Δt = Δt, stop_time = 10, minimum_relative_step = 1e-10)

time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05, min_Δt=1e-4, max_Δt=1)
simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities
d = CenterField(grid)

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

function progress(sim)
    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    msg = @sprintf("Iter: %d, time: %6.3e, Δt: %6.3e, Poisson iters: %d",
                    iteration(sim), time(sim), sim.Δt, pressure_iters)

    compute_flow_divergence!(d, sim.model)

    msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max d: %6.3e, max pressure: %6.3e, mean pressure: %6.3e",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.v),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
                    mean(sim.model.pressures.pNHS),
    )

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(
    progress,
    IterationInterval(1),
)

u, v, w = model.velocities
c = model.tracers.c
compute_flow_divergence!(d, model)
p = model.pressures.pNHS

output_fields = (; u, v, w, c, d, p)

simulation.output_writers[:output_writer] = JLD2Writer(
    model,
    output_fields,
    filename = "vortex_sheet",
    schedule = TimeInterval(Δt * 50),
    overwrite_existing = true,
)

run!(simulation)
#%%
u_data = FieldTimeSeries("./vortex_sheet.jld2", "u")
w_data = FieldTimeSeries("./vortex_sheet.jld2", "w")
c_data = FieldTimeSeries("./vortex_sheet.jld2", "c")
d_data = FieldTimeSeries("./vortex_sheet.jld2", "d")
p_data = FieldTimeSeries("./vortex_sheet.jld2", "p")
#%%
times = u_data.times
Nt = length(times)

xC = u_data.grid.underlying_grid.xᶜᵃᵃ[1:Nx]
xF = u_data.grid.underlying_grid.xᶠᵃᵃ[1:Nx+1]
zC = u_data.grid.z.cᵃᵃᶜ[1:Nz]
zF = u_data.grid.z.cᵃᵃᶠ[1:Nz+1]

ulim = (minimum(interior(u_data)), maximum(interior(u_data)))
wlim = (-maximum(abs, interior(w_data)), maximum(abs, interior(w_data)))
clim = (minimum(interior(c_data)), maximum(interior(c_data)))
# dlim = (-maximum(abs, interior(d_data)), maximum(abs, interior(d_data)))
dlim = (-2e-9, 2e-9)
plim = (-maximum(abs, interior(p_data)), maximum(abs, interior(p_data))) ./ 2
#%%
using CairoMakie
fig = Figure(size=(1200, 1000))
axu = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", title="u", aspect=DataAspect())
axw = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", title="w", aspect=DataAspect())
axc = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)", title="c", aspect=DataAspect())
axd = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)", title="Divergence, lims = $(dlim)", aspect=DataAspect())
axp = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)", title="Pressure", aspect=DataAspect())

n = Observable(Nt)

uₙ = @lift interior(u_data[$n], :, 1, :)
wₙ = @lift interior(w_data[$n], :, 1, :)
cₙ = @lift interior(c_data[$n], :, 1, :)
dₙ = @lift interior(d_data[$n], :, 1, :)
pₙ = @lift interior(p_data[$n], :, 1, :)
timeₙ = @lift "Time = $(times[$n])"

heatmap!(axu, xF, zC, uₙ, colormap=:turbo, colorrange=ulim)
heatmap!(axw, xC, zF, wₙ, colormap=:balance, colorrange=wlim)
heatmap!(axc, xC, zC, cₙ, colormap=:turbo, colorrange=clim)
heatmap!(axd, xC, zC, dₙ, colormap=:balance, colorrange=dlim)
heatmap!(axp, xC, zC, pₙ, colormap=:balance, colorrange=plim)

Label(fig[0, :], timeₙ, tellwidth=false)
display(fig)

CairoMakie.record(fig, "./vortex_sheet.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
    n[] = nn
end
#%%