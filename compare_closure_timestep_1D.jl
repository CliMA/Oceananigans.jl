using Oceananigans
include("NN_closure_global_Ri_nof_BBLRifirstzone510_train62newstrongSO_20seed_Ri8020_round3.jl")
include("xin_kai_vertical_diffusivity_local_2step_train56newstrongSO.jl")

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: TKEDissipationVerticalDiffusivity, CATKEVerticalDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10
using CairoMakie

model_architecture = CPU()

const Nz = 32
const Lz = 256

const dTdz = 0.015
const dSdz = 0.002

const T_surface = 20
const S_surface = 37
const Qᵀ = 0.0002
const Qˢ = -2.0e-5
const Qᵁ = -0.0001
const f₀ = 0

T_initial(z) = dTdz * z + T_surface
S_initial(z) = dSdz * z + S_surface

nn_closure = (XinKaiLocalVerticalDiffusivity(), NNFluxClosure(model_architecture))
function CATKE_ocean_closure()
    mixing_length = CATKEMixingLength(Cᵇ=0.28)
    turbulent_kinetic_energy_equation = CATKEEquation()
    return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation)
  end
CATKE_closure = CATKE_ocean_closure()
kϵ_closure = TKEDissipationVerticalDiffusivity()

function setup_model(closure)
    grid = RectilinearGrid(CPU(),
                           topology = (Flat, Flat, Bounded),
                           size = Nz,
                           halo = 3,
                           z = (-Lz, 0))
    
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ))
    S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ))

    coriolis = FPlane(f=f₀)

    if closure isa CATKEVerticalDiffusivity
        tracers = (:T, :S, :e)
    elseif closure isa TKEDissipationVerticalDiffusivity
        tracers = (:T, :S, :e, :ϵ)
    else
        tracers = (:T, :S)
    end

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        free_surface = ImplicitFreeSurface(),
                                        momentum_advection = WENO(grid = grid),
                                        tracer_advection = WENO(grid = grid),
                                        buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
                                        coriolis = coriolis,
                                        closure = closure,
                                        tracers = tracers,
                                        boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs))

    noise(z) = rand() * exp(z / 8)

    T_initial_noisy(z) = T_initial(z) + 1e-6 * noise(z)
    S_initial_noisy(z) = S_initial(z) + 1e-6 * noise(z)
    
    set!(model, T=T_initial_noisy, S=S_initial_noisy)
    update_state!(model)

    return model
end

function run_simulation(model, Δt)
    stop_time = 4days
    simulation = Simulation(model, Δt = Δt, stop_time = stop_time)

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

    simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))
    
    u, v, w = model.velocities
    T, S = model.tracers.T, model.tracers.S

    ubar = Field(Average(u, dims = (1,2)))
    vbar = Field(Average(v, dims = (1,2)))
    Tbar = Field(Average(T, dims = (1,2)))
    Sbar = Field(Average(S, dims = (1,2)))

    averaged_outputs = (; ubar, vbar, Tbar, Sbar)

    if model.closure isa CATKEVerticalDiffusivity
        OUTPUT_PATH = "./benchmark/CATKE_timestep"
    elseif model.closure isa TKEDissipationVerticalDiffusivity
        OUTPUT_PATH = "./benchmark/kepsilon_timestep"
    else
        OUTPUT_PATH = "./benchmark/NN_timestep"
    end

    mkpath(OUTPUT_PATH)
    simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
                                                        filename = "$(OUTPUT_PATH)/dt_$(Δt)_dTdz_$(dTdz)_dSdz_$(dSdz)_QT_$(Qᵀ)_QS_$(Qˢ)_QU_$(Qᵁ)_f_$(f₀).jld2",
                                                        schedule = TimeInterval(240minutes),
                                                        overwrite_existing = true)

    run!(simulation)
end

setup_and_run_simulation(closure, Δt) = run_simulation(setup_model(closure), Δt)

closures = [nn_closure, CATKE_closure, kϵ_closure]
Δts = [1minute, 5minutes, 15minutes, 30minutes, 60minutes, 120minutes, 240minutes]
for (i, closure) in enumerate(closures), Δt in Δts
    @info "Running simulation with closure $i and Δt: $Δt"
    setup_and_run_simulation(closure, Δt)
end

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

#%%
dir_type = "CATKE"
FILE_DIR = "./benchmark/$(dir_type)_timestep"
ubar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "ubar") for Δt in Δts]
vbar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "vbar") for Δt in Δts]
Tbar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "Tbar") for Δt in Δts]
Sbar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "Sbar") for Δt in Δts]

#%%
zC = znodes(Tbar_datas[1].grid, Center())
zF = znodes(Tbar_datas[1].grid, Face())

Nt = length(Tbar_datas[1].times)

fig = Figure(size = (1300, 500))
axu = CairoMakie.Axis(fig[1, 1], xlabel = "u (m s⁻¹)", ylabel = "z (m)")
axv = CairoMakie.Axis(fig[1, 2], xlabel = "v (m s⁻¹)", ylabel = "z (m)")
axT = CairoMakie.Axis(fig[1, 3], xlabel = "T (°C)", ylabel = "z (m)")
axS = CairoMakie.Axis(fig[1, 4], xlabel = "S (g kg⁻¹)", ylabel = "z (m)")

n = Observable(1)

ubarₙs = [@lift interior(ubar_data[$n], 1, 1, :) for ubar_data in ubar_datas]
vbarₙs = [@lift interior(vbar_data[$n], 1, 1, :) for vbar_data in vbar_datas]
Tbarₙs = [@lift interior(Tbar_data[$n], 1, 1, :) for Tbar_data in Tbar_datas]
Sbarₙs = [@lift interior(Sbar_data[$n], 1, 1, :) for Sbar_data in Sbar_datas]

title_str = @lift "Time: $(round(Tbar_datas[1].times[$n] / 86400, digits=3)) days"

ulim = (find_min([interior(ubar_data) for ubar_data in ubar_datas[1:end-1]]...), find_max([interior(ubar_data) for ubar_data in ubar_datas[1:end-1]]...))
vlim = (find_min([interior(vbar_data) for vbar_data in vbar_datas[1:end-1]]...) - 1e-4, find_max([interior(vbar_data) for vbar_data in vbar_datas[1:end-1]]...) + 1e-4)
Tlim = (find_min([interior(Tbar_data) for Tbar_data in Tbar_datas]...), find_max([interior(Tbar_data) for Tbar_data in Tbar_datas]...))
Slim = (find_min([interior(Sbar_data) for Sbar_data in Sbar_datas]...), find_max([interior(Sbar_data) for Sbar_data in Sbar_datas]...))

for (i, Δt) in enumerate(Δts)
    lines!(axu, ubarₙs[i], zC, label = "Δt = $Δt", linewidth=2)
    lines!(axv, vbarₙs[i], zC, label = "Δt = $Δt", linewidth=2)
    lines!(axT, Tbarₙs[i], zC, label = "Δt = $Δt", linewidth=2)
    lines!(axS, Sbarₙs[i], zC, label = "Δt = $Δt", linewidth=2)
end

xlims!(axu, ulim)
xlims!(axv, vlim)
xlims!(axT, Tlim)
xlims!(axS, Slim)

Legend(fig[2, :], axu, position = :lb, orientation=:horizontal)
Label(fig[0, :], title_str, tellwidth = false)

linkyaxes!(axu, axv, axT, axS)

hideydecorations!(axv, ticks=false, grid=false)
hideydecorations!(axT, ticks=false, grid=false)
hideydecorations!(axS, ticks=false, grid=false)
display(fig)

CairoMakie.record(fig, "./Output/$(dir_type)_timesteps_test.mp4", 1:Nt, framerate=3, px_per_unit=2) do nn
    n[] = nn
end
#%%