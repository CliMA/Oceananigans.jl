using Oceananigans
using Printf
using Statistics
using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10
# using CairoMakie

model_architecture = GPU()

const Nx = 192
const Ny = 192
const Nz = 384

const Lx = 512
const Ly = 512

const Δx = Lx / Nx
const Δy = Ly / Ny
const Δz = 2

const Lz = Nz * Δz

const dTdz = (30 - 10) / 1024 / 2
const dSdz = (37 - 34) / 1024 / 2

const T_surface = 30
const S_surface = 37
const Qᵀ = 0.0002
const Qˢ = -2.0e-5
const Qᵁ = -0.0001
const f₀ = 1e-4
const T_period = 1day
const S_period = 2.63158day

const eos = TEOS10EquationOfState()

T_initial(z) = dTdz * z + T_surface
S_initial(z) = dSdz * z + S_surface

grid = RectilinearGrid(GPU(),
                        topology = (Periodic, Periodic, Bounded),
                        size = (Nx, Ny, Nz),
                        halo = (5, 5, 5),
                        x = (0, Lx),
                        y = (0, Ly),
                        z = (-Lz, 0))

@inline T_flux(x, y, t) = Qᵀ * cos(2π / T_period * t) + Qᵀ / 2
@inline S_flux(x, y, t) = Qˢ * cos(2π / S_period * t)

u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))
T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(T_flux), bottom=GradientBoundaryCondition(dTdz))
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(S_flux), bottom=GradientBoundaryCondition(dSdz))

damping_rate = 1/15minute

T_target(x, y, z, t) = T_initial(z)
S_target(x, y, z, t) = S_initial(z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
T_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=T_target)
S_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=S_target)

coriolis = FPlane(f=f₀)

model = NonhydrostaticModel(grid = grid,
                            advection = WENO(order=9),
                            buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
                            coriolis = coriolis,
                            closure = nothing,
                            tracers = (:T, :S),
                            boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs),
                            forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, T=T_sponge, S=S_sponge))

const ρ₀ = eos.reference_density
const g = model.buoyancy.model.gravitational_acceleration

noise(z) = rand() * exp(z / 8)

T_initial_noisy(x, y, z) = T_initial(z) + 1e-6 * noise(z)
S_initial_noisy(x, y, z) = S_initial(z) + 1e-6 * noise(z)

set!(model, T=T_initial_noisy, S=S_initial_noisy)

Δt = 1
stop_time = 60days
simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

wizard = TimeStepWizard(max_change=1.05, max_Δt=5minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.tracers.T),
        maximum(abs, sim.model.tracers.S),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

@inline function get_buoyancy(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
    @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
    ρ′ = ρ - ρ₀
    return -g * ρ′ / ρ₀
  end
  
@inline function get_density(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
    @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
    return ρ
end

b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy, model.grid, model.buoyancy, model.tracers)
b = Field(b_op)
compute!(b)

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid, model.buoyancy, model.tracers)
ρ = Field(ρ_op)
compute!(ρ)

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(200))
    
u, v, w = model.velocities
T, S = model.tracers.T, model.tracers.S

ubar = Field(Average(u, dims = (1,2)))
vbar = Field(Average(v, dims = (1,2)))
Tbar = Field(Average(T, dims = (1,2)))
Sbar = Field(Average(S, dims = (1,2)))
bbar = Field(Average(b, dims = (1,2)))
ρbar = Field(Average(ρ, dims = (1,2)))

uw = Field(Average(u * w, dims = (1,2)))
vw = Field(Average(v * w, dims = (1,2)))
wb = Field(Average(w * b, dims = (1,2)))
wT = Field(Average(w * T, dims = (1,2)))
wS = Field(Average(w * S, dims = (1,2)))
wρ = Field(Average(w * ρ, dims = (1,2)))

outputs = (; u, v, w, T, S)
averaged_outputs = (; ubar, vbar, Tbar, Sbar, bbar, ρbar, uw, vw, wb, wT, wS, wρ)

OUTPUT_PATH = "./Output/LES_60days/"
mkpath(OUTPUT_PATH)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(OUTPUT_PATH)/aperiodic_T_$(T_period)_S_$(S_period)_dt_$(Δt)_dTdz_$(dTdz)_dSdz_$(dSdz)_QT_$(Qᵀ)_QS_$(Qˢ)_QU_$(Qᵁ)_f_$(f₀)_fields.jld2",
                                                    schedule = TimeInterval(2days))
                                                    # overwrite_existing = true)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
                                                    filename = "$(OUTPUT_PATH)/aperiodic_T_$(T_period)_S_$(S_period)_dt_$(Δt)_dTdz_$(dTdz)_dSdz_$(dSdz)_QT_$(Qᵀ)_QS_$(Qˢ)_QU_$(Qᵁ)_f_$(f₀)_averaged_fields.jld2",
                                                    schedule = TimeInterval(120minutes))
                                                    # overwrite_existing = true)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1days), prefix="$(OUTPUT_PATH)/model_checkpoint")

files = readdir(OUTPUT_PATH)
checkpoint_files = files[occursin.("model_checkpoint_iteration", files)]
if !isempty(checkpoint_files)
    checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
    pickup_iter = maximum(checkpoint_iters)
    run!(simulation, pickup="$(OUTPUT_PATH)/model_checkpoint_iteration$(pickup_iter).jld2")
else
    run!(simulation)
    # @info "no checkpointer files found"
end