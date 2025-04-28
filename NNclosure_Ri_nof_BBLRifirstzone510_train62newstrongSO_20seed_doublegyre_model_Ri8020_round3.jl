#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
include("NN_closure_global_Ri_nof_BBLRifirstzone510_train62newstrongSO_20seed_Ri8020_round3.jl")
include("xin_kai_vertical_diffusivity_local_2step_train56newstrongSO.jl")

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics
using CairoMakie

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10
using ColorSchemes

#%%
filename = "doublegyre_30Cwarmflushbottom10_relaxation_8days_zWENO5_NN_closure_NDE5_Ri_BBLRifirztzone510train62newstrongSO_20seed_round3_Ri8020_100years_threshold24"
FILE_DIR = "./Output/$(filename)"
# FILE_DIR = "/storage6/xinkai/NN_Oceananigans/$(filename)"
@info "$(FILE_DIR)"
mkpath(FILE_DIR)

# Architecture
model_architecture = GPU()

nn_closure = NNFluxClosure(model_architecture)
base_closure = XinKaiLocalVerticalDiffusivity()
closure = (base_closure, nn_closure)

advection_scheme = FluxFormAdvection(WENO(order=5), WENO(order=5), WENO(order=5))

# number of grid points
const Nx = 100
const Ny = 100
const Nz = 200

const Δz = 8meters
const Lx = 4000kilometers
const Ly = 6000kilometers
const Lz = Nz * Δz

grid = RectilinearGrid(model_architecture, Float64,
                       topology = (Bounded, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (4, 4, 4),
                          x = (-Lx/2, Lx/2),
                          y = (-Ly/2, Ly/2),
                          z = (-Lz, 0))

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####
const T_north = 0
const T_south = 30
const T_mid = (T_north + T_south) / 2
const ΔT = T_south - T_north

const S_north = 34
const S_south = 37
const S_mid = (S_north + S_south) / 2

const τ₀ = 1e-4

const μ_drag = 1/30days
const μ_T = 1/8days
#####
##### Forcing and initial condition
#####
# @inline T_initial(x, y, z) = T_north + ΔT / 2 * (1 + z / Lz)
# @inline T_initial(x, y, z) = (T_north + T_south / 2) + 5 * (1 + z / Lz)
@inline T_initial(x, y, z) = 10 + 20 * (1 + z / Lz)

@inline surface_u_flux(x, y, t) = -τ₀ * cos(2π * y / Ly)

surface_u_flux_bc = FluxBoundaryCondition(surface_u_flux)

@inline u_drag(x, y, t, u) = @inbounds -μ_drag * Lz * u
@inline v_drag(x, y, t, v) = @inbounds -μ_drag * Lz * v

u_drag_bc  = FluxBoundaryCondition(u_drag; field_dependencies=:u)
v_drag_bc  = FluxBoundaryCondition(v_drag; field_dependencies=:v)

u_bcs = FieldBoundaryConditions(   top = surface_u_flux_bc, 
                                bottom = u_drag_bc,
                                 north = ValueBoundaryCondition(0),
                                 south = ValueBoundaryCondition(0))

v_bcs = FieldBoundaryConditions(   top = FluxBoundaryCondition(0),
                                bottom = v_drag_bc,
                                  east = ValueBoundaryCondition(0),
                                  west = ValueBoundaryCondition(0))

@inline T_ref(y) = T_mid - ΔT / Ly * y
@inline surface_T_flux(x, y, t, T) = μ_T * Δz * (T - T_ref(y))
surface_T_flux_bc = FluxBoundaryCondition(surface_T_flux; field_dependencies=:T)
T_bcs = FieldBoundaryConditions(top = surface_T_flux_bc)

@inline S_ref(y) = (S_north - S_south) / Ly * y + S_mid
@inline S_initial(x, y, z) = S_ref(y)
@inline surface_S_flux(x, y, t, S) = μ_T * Δz * (S - S_ref(y))
surface_S_flux_bc = FluxBoundaryCondition(surface_S_flux; field_dependencies=:S)
S_bcs = FieldBoundaryConditions(top = surface_S_flux_bc)

#####
##### Coriolis
#####
coriolis = BetaPlane(rotation_rate=7.292115e-5, latitude=45, radius=6371e3)

#####
##### Model building
#####

@info "Building a model..."

# This is a weird bug. If a model is not initialized with a closure other than XinKaiVerticalDiffusivity,
# the code will throw a CUDA: illegal memory access error for models larger than a small size.
# This is a workaround to initialize the model with a closure other than XinKaiVerticalDiffusivity first,
# then the code will run without any issues.
model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
    momentum_advection = advection_scheme,
    tracer_advection = advection_scheme,
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = VerticalScalarDiffusivity(ν=1e-5, κ=1e-5),
    tracers = (:T, :S),
    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
)

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
    momentum_advection = advection_scheme,
    tracer_advection = advection_scheme,
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = closure,
    tracers = (:T, :S),
    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
)

@info "Built $model."

noise(z) = rand() * exp(z / 8)

T_initial_noisy(x, y, z) = T_initial(x, y, z) + 1e-6 * noise(z)
S_initial_noisy(x, y, z) = S_initial(x, y, z) + 1e-6 * noise(z)

set!(model, T=T_initial_noisy, S=S_initial_noisy)
using Oceananigans.TimeSteppers: update_state!
update_state!(model)
#####
##### Simulation building
#####
Δt₀ = 5minutes
stop_time = 72000days
# stop_time = 1080days

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

# add timestep wizard callback
# wizard = TimeStepWizard(cfl=0.25, max_change=1.05, max_Δt=12minutes)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): %6.3e, max(v): %6.3e, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.velocities.u),
        maximum(abs, sim.model.velocities.v),
        maximum(abs, sim.model.tracers.T),
        maximum(abs, sim.model.tracers.S),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

#####
##### Diagnostics
#####
u, v, w = model.velocities
T, S = model.tracers.T, model.tracers.S
U_bt = Field(Integral(u, dims=3));
Ψ = Field(CumulativeIntegral(-U_bt, dims=2));
first_index = model.diffusivity_fields[2].first_index
last_index = model.diffusivity_fields[2].last_index
wT_NN = model.diffusivity_fields[2].wT
wS_NN = model.diffusivity_fields[2].wS

Ri = model.diffusivity_fields[1].Ri
κ = model.diffusivity_fields[1].κᶜ
wT_base = κ * ∂z(T)
wS_base = κ * ∂z(S)

wT = wT_NN + wT_base
wS = wS_NN + wS_base

@inline function get_N²(i, j, k, grid, b, C)
  return ∂z_b(i, j, k, grid, b, C)
end

N²_op = KernelFunctionOperation{Center, Center, Face}(get_N², model.grid, model.buoyancy.model, model.tracers)
N² = Field(N²_op)

@inline function get_density(i, j, k, grid, b, C)
  T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
  @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, b.model.equation_of_state)
  return ρ
end

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid, model.buoyancy, model.tracers)
ρ = Field(ρ_op)

@inline function get_top_buoyancy_flux(i, j, k, grid, buoyancy, T_bc, S_bc, velocities, tracers, clock)
  return top_buoyancy_flux(i, j, grid, buoyancy, (; T=T_bc, S=S_bc), clock, merge(velocities, tracers))
end

Qb = KernelFunctionOperation{Center, Center, Nothing}(get_top_buoyancy_flux, model.grid, model.buoyancy, T.boundary_conditions.top, S.boundary_conditions.top, model.velocities, model.tracers, model.clock)
Qb = Field(Qb)

ubar_zonal = Average(u, dims=1)
vbar_zonal = Average(v, dims=1)
wbar_zonal = Average(w, dims=1)
Tbar_zonal = Average(T, dims=1)
Sbar_zonal = Average(S, dims=1)
ρbar_zonal = Average(ρ, dims=1)

wT_NNbar_zonal = Average(wT_NN, dims=1)
wS_NNbar_zonal = Average(wS_NN, dims=1)

wT_basebar_zonal = Average(wT_base, dims=1)
wS_basebar_zonal = Average(wS_base, dims=1)

wTbar_zonal = Average(wT, dims=1)
wSbar_zonal = Average(wS, dims=1)

outputs = (; u, v, w, T, S, ρ, N², wT_NN, wS_NN, wT_base, wS_base, wT, wS, Ri)
zonal_outputs = (; ubar_zonal, vbar_zonal, wbar_zonal, Tbar_zonal, Sbar_zonal, ρbar_zonal, wT_NNbar_zonal, wS_NNbar_zonal, wT_basebar_zonal, wS_basebar_zonal, wTbar_zonal, wSbar_zonal)

#####
##### Build checkpointer and output writer
#####
simulation.output_writers[:xy] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy",
                                                    indices = (:, :, Nz),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_190] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_190",
                                                    indices = (:, :, 190),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_180] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_180",
                                                    indices = (:, :, 180),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_170] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_170",
                                                    indices = (:, :, 170),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_160] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_160",
                                                    indices = (:, :, 160),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_150] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_150",
                                                    indices = (:, :, 150),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_140] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_140",
                                                    indices = (:, :, 140),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_130] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_130",
                                                    indices = (:, :, 130),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_120] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_120",
                                                    indices = (:, :, 120),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_110] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_110",
                                                    indices = (:, :, 110),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xy_100] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy_100",
                                                    indices = (:, :, 100),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz",
                                                    indices = (1, :, :),
                                                    schedule = TimeInterval(10days))
                                                    
simulation.output_writers[:xz] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz",
                                                    indices = (:, 1, :),
                                                    schedule = TimeInterval(10days))
                                                    
simulation.output_writers[:yz_10] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_10",
                                                    indices = (10, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_20] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_20",
                                                    indices = (20, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_30] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_30",
                                                    indices = (30, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_40] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_40",
                                                    indices = (40, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_50] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_50",
                                                    indices = (50, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_60] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_60",
                                                    indices = (60, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_70] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_70",
                                                    indices = (70, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_80] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_80",
                                                    indices = (80, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_90] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_90",
                                                    indices = (90, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_100] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_100",
                                                    indices = (100, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_5] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_5",
                                                    indices = (:, 5, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_15] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_15",
                                                    indices = (:, 15, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_25] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_25",
                                                    indices = (:, 25, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_35] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_35",
                                                    indices = (:, 35, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_45] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_45",
                                                    indices = (:, 45, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_55] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_55",
                                                    indices = (:, 55, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_65] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_65",
                                                    indices = (:, 65, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_75] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_75",
                                                    indices = (:, 75, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_85] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_85",
                                                    indices = (:, 85, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:xz_95] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz_95",
                                                    indices = (:, 95, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:zonal_average] = JLD2OutputWriter(model, zonal_outputs,
                                                             filename = "$(FILE_DIR)/averaged_fields_zonal",
                                                             schedule = TimeInterval(10days))

simulation.output_writers[:BBL] = JLD2OutputWriter(model, (; first_index, last_index, Qb),
                                                    filename = "$(FILE_DIR)/instantaneous_fields_NN_active_diagnostics",
                                                    indices = (:, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:streamfunction] = JLD2OutputWriter(model, (; Ψ=Ψ,),
                                                    filename = "$(FILE_DIR)/averaged_fields_streamfunction",
                                                    schedule = AveragedTimeInterval(1800days, window=1800days))

simulation.output_writers[:streamfunction_10] = JLD2OutputWriter(model, (; Ψ=Ψ,),
                                                    filename = "$(FILE_DIR)/averaged_fields_streamfunction_10years",
                                                    schedule = AveragedTimeInterval(3600days, window=3600days))

simulation.output_writers[:complete_fields] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields",
                                                    schedule = TimeInterval(1800days))

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                    schedule = TimeInterval(18000days),
                                                    prefix = "$(FILE_DIR)/checkpointer")

@info "Running the simulation..."

try
  files = readdir(FILE_DIR)
  checkpoint_files = files[occursin.("checkpointer_iteration", files)]
  if !isempty(checkpoint_files)
      checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
      pickup_iter = maximum(checkpoint_iters)
      run!(simulation, pickup="$(FILE_DIR)/checkpointer_iteration$(pickup_iter).jld2")
  else
      run!(simulation)
  end
catch err
  @info "run! threw an error! The error message is"
  showerror(stdout, err)
end

#%%
T_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "T")
T_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "T")
T_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "T")

S_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "S")
S_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "S")
S_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "S")

u_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "u")
u_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "u")
u_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "u")

v_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "v")
v_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "v")
v_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "v")

times = T_xy_data.times ./ 24 ./ 60^2
Nt = length(times)
timeframes = 1:Nt

# Nx, Ny, Nz = T_xy_data.grid.Nx, T_xy_data.grid.Ny, T_xy_data.grid.Nz
xC, yC, zC = T_xy_data.grid.xᶜᵃᵃ[1:Nx], T_xy_data.grid.yᵃᶜᵃ[1:Ny], T_xy_data.grid.zᵃᵃᶜ[1:Nz]
zF = T_xy_data.grid.zᵃᵃᶠ[1:Nz+1]

# Lx, Ly, Lz = T_xy_data.grid.Lx, T_xy_data.grid.Ly, T_xy_data.grid.Lz

xCs_xy = xC
yCs_xy = yC
zCs_xy = [zC[Nz] for x in xCs_xy, y in yCs_xy]

yCs_yz = yC
xCs_yz = range(xC[1], stop=xC[1], length=length(zC))
zCs_yz = zeros(length(xCs_yz), length(yCs_yz))
for j in axes(zCs_yz, 2)
  zCs_yz[:, j] .= zC
end

xCs_xz = xC
yCs_xz = range(yC[1], stop=yC[1], length=length(zC))
zCs_xz = zeros(length(xCs_xz), length(yCs_xz))
for i in axes(zCs_xz, 1)
  zCs_xz[i, :] .= zC
end

xFs_xy = xC
yFs_xy = yC
zFs_xy = [zF[Nz+1] for x in xFs_xy, y in yFs_xy]

yFs_yz = yC
xFs_yz = range(xC[1], stop=xC[1], length=length(zF))
zFs_yz = zeros(length(xFs_yz), length(yFs_yz))
for j in axes(zFs_yz, 2)
  zFs_yz[:, j] .= zF
end

xFs_xz = xC
yFs_xz = range(yC[1], stop=yC[1], length=length(zF))
zFs_xz = zeros(length(xFs_xz), length(yFs_xz))
for i in axes(zFs_xz, 1)
  zFs_xz[i, :] .= zF
end

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

# for freeconvection
# startheight = 64

# for wind mixing
startheight = 1
Tlim = (find_min(interior(T_xy_data, :, :, 1, timeframes), interior(T_yz_data, 1, :, startheight:Nz, timeframes), interior(T_xz_data, :, 1, startheight:Nz, timeframes)), 
        find_max(interior(T_xy_data, :, :, 1, timeframes), interior(T_yz_data, 1, :, startheight:Nz, timeframes), interior(T_xz_data, :, 1, startheight:Nz, timeframes)))
Slim = (find_min(interior(S_xy_data, :, :, 1, timeframes), interior(S_yz_data, 1, :, startheight:Nz, timeframes), interior(S_xz_data, :, 1, startheight:Nz, timeframes)), 
        find_max(interior(S_xy_data, :, :, 1, timeframes), interior(S_yz_data, 1, :, startheight:Nz, timeframes), interior(S_xz_data, :, 1, startheight:Nz, timeframes)))
ulim = (-find_max(interior(u_xy_data, :, :, 1, timeframes), interior(u_yz_data, 1, :, startheight:Nz, timeframes), interior(u_xz_data, :, 1, startheight:Nz, timeframes)),
         find_max(interior(u_xy_data, :, :, 1, timeframes), interior(u_yz_data, 1, :, startheight:Nz, timeframes), interior(u_xz_data, :, 1, startheight:Nz, timeframes)))
vlim = (-find_max(interior(v_xy_data, :, :, 1, timeframes), interior(v_yz_data, 1, :, startheight:Nz, timeframes), interior(v_xz_data, :, 1, startheight:Nz, timeframes)),
         find_max(interior(v_xy_data, :, :, 1, timeframes), interior(v_yz_data, 1, :, startheight:Nz, timeframes), interior(v_xz_data, :, 1, startheight:Nz, timeframes)))

colorscheme = colorschemes[:balance]
T_colormap = colorscheme
S_colormap = colorscheme
u_colormap = colorscheme
v_colormap = colorscheme

T_color_range = Tlim
S_color_range = Slim
u_color_range = ulim
v_color_range = vlim
#%%
plot_aspect = (2, 3, 0.5)
fig = Figure(size=(1500, 700))
axT = CairoMakie.Axis3(fig[1, 1], title="Temperature (°C)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axS = CairoMakie.Axis3(fig[1, 3], title="Salinity (g kg⁻¹)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axu = CairoMakie.Axis3(fig[2, 1], title="u (m/s)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axv = CairoMakie.Axis3(fig[2, 3], title="v (m/s)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)

n = Observable(1)

T_xy = @lift interior(T_xy_data[$n], :, :, 1)
T_yz = @lift transpose(interior(T_yz_data[$n], 1, :, :))
T_xz = @lift interior(T_xz_data[$n], :, 1, :)

S_xy = @lift interior(S_xy_data[$n], :, :, 1)
S_yz = @lift transpose(interior(S_yz_data[$n], 1, :, :))
S_xz = @lift interior(S_xz_data[$n], :, 1, :)

u_xy = @lift interior(u_xy_data[$n], :, :, 1)
u_yz = @lift transpose(interior(u_yz_data[$n], 1, :, :))
u_xz = @lift interior(u_xz_data[$n], :, 1, :)

v_xy = @lift interior(v_xy_data[$n], :, :, 1)
v_yz = @lift transpose(interior(v_yz_data[$n], 1, :, :))
v_xz = @lift interior(v_xz_data[$n], :, 1, :)

# time_str = @lift "Surface Cooling, Time = $(round(times[$n], digits=2)) hours"
time_str = @lift "Surface Wind Stress, Time = $(round(times[$n], digits=2)) days"
Label(fig[0, :], text=time_str, tellwidth=false, font=:bold)

T_xy_surface = surface!(axT, xCs_xy, yCs_xy, zCs_xy, color=T_xy, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
T_yz_surface = surface!(axT, xCs_yz, yCs_yz, zCs_yz, color=T_yz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
T_xz_surface = surface!(axT, xCs_xz, yCs_xz, zCs_xz, color=T_xz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])

S_xy_surface = surface!(axS, xCs_xy, yCs_xy, zCs_xy, color=S_xy, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
S_yz_surface = surface!(axS, xCs_yz, yCs_yz, zCs_yz, color=S_yz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
S_xz_surface = surface!(axS, xCs_xz, yCs_xz, zCs_xz, color=S_xz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])

u_xy_surface = surface!(axu, xCs_xy, yCs_xy, zCs_xy, color=u_xy, colormap=u_colormap, colorrange = u_color_range, lowclip=u_colormap[1], highclip=u_colormap[end])
u_yz_surface = surface!(axu, xCs_yz, yCs_yz, zCs_yz, color=u_yz, colormap=u_colormap, colorrange = u_color_range, lowclip=u_colormap[1], highclip=u_colormap[end])
u_xz_surface = surface!(axu, xCs_xz, yCs_xz, zCs_xz, color=u_xz, colormap=u_colormap, colorrange = u_color_range, lowclip=u_colormap[1], highclip=u_colormap[end])

v_xy_surface = surface!(axv, xCs_xy, yCs_xy, zCs_xy, color=v_xy, colormap=v_colormap, colorrange = v_color_range, lowclip=v_colormap[1], highclip=v_colormap[end])
v_yz_surface = surface!(axv, xCs_yz, yCs_yz, zCs_yz, color=v_yz, colormap=v_colormap, colorrange = v_color_range, lowclip=v_colormap[1], highclip=v_colormap[end])
v_xz_surface = surface!(axv, xCs_xz, yCs_xz, zCs_xz, color=v_xz, colormap=v_colormap, colorrange = v_color_range, lowclip=v_colormap[1], highclip=v_colormap[end])

Colorbar(fig[1,2], T_xy_surface)
Colorbar(fig[1,4], S_xy_surface)
Colorbar(fig[2,2], u_xy_surface)
Colorbar(fig[2,4], v_xy_surface)

xlims!(axT, (-Lx/2, Lx/2))
xlims!(axS, (-Lx/2, Lx/2))
xlims!(axu, (-Lx/2, Lx/2))
xlims!(axv, (-Lx/2, Lx/2))

ylims!(axT, (-Ly/2, Ly/2))
ylims!(axS, (-Ly/2, Ly/2))
ylims!(axu, (-Ly/2, Ly/2))
ylims!(axv, (-Ly/2, Ly/2))

zlims!(axT, (-Lz, 0))
zlims!(axS, (-Lz, 0))
zlims!(axu, (-Lz, 0))
zlims!(axv, (-Lz, 0))

@info "Recording 3D fields"
CairoMakie.record(fig, "$(FILE_DIR)/$(filename)_3D_instantaneous_fields.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
    n[] = nn
end

#%%
@info "Recording T fields and fluxes in yz"

fieldname = "T"
fluxname = "wT_NN"
field_NN_data_00 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", fieldname, backend=OnDisk())
field_NN_data_10 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_10.jld2", fieldname, backend=OnDisk())
field_NN_data_20 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_20.jld2", fieldname, backend=OnDisk())
field_NN_data_30 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_30.jld2", fieldname, backend=OnDisk())
field_NN_data_40 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_40.jld2", fieldname, backend=OnDisk())
field_NN_data_50 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_50.jld2", fieldname, backend=OnDisk())
field_NN_data_60 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_60.jld2", fieldname, backend=OnDisk())
field_NN_data_70 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_70.jld2", fieldname, backend=OnDisk())
field_NN_data_80 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_80.jld2", fieldname, backend=OnDisk())
field_NN_data_90 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_90.jld2", fieldname, backend=OnDisk())

flux_NN_data_00 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", fluxname, backend=OnDisk())
flux_NN_data_10 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_10.jld2", fluxname, backend=OnDisk())
flux_NN_data_20 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_20.jld2", fluxname, backend=OnDisk())
flux_NN_data_30 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_30.jld2", fluxname, backend=OnDisk())
flux_NN_data_40 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_40.jld2", fluxname, backend=OnDisk())
flux_NN_data_50 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_50.jld2", fluxname, backend=OnDisk())
flux_NN_data_60 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_60.jld2", fluxname, backend=OnDisk())
flux_NN_data_70 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_70.jld2", fluxname, backend=OnDisk())
flux_NN_data_80 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_80.jld2", fluxname, backend=OnDisk())
flux_NN_data_90 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_90.jld2", fluxname, backend=OnDisk())

xC = field_NN_data_00.grid.xᶜᵃᵃ[1:field_NN_data_00.grid.Nx]
yC = field_NN_data_00.grid.yᵃᶜᵃ[1:field_NN_data_00.grid.Ny]
zC = field_NN_data_00.grid.zᵃᵃᶜ[1:field_NN_data_00.grid.Nz]
zF = field_NN_data_00.grid.zᵃᵃᶠ[1:field_NN_data_00.grid.Nz+1]

Nt = length(field_NN_data_90)
times = field_NN_data_00.times / 24 / 60^2 / 365
timeframes = 1:Nt

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

#%%
fig = Figure(size=(3000, 1200))

axfield_00 = CairoMakie.Axis(fig[1, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_00.grid.xᶜᵃᵃ[field_NN_data_00.indices[1][1]] / 1000) km")
axfield_10 = CairoMakie.Axis(fig[1, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_10.grid.xᶜᵃᵃ[field_NN_data_10.indices[1][1]] / 1000) km")
axfield_20 = CairoMakie.Axis(fig[2, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_20.grid.xᶜᵃᵃ[field_NN_data_20.indices[1][1]] / 1000) km")
axfield_30 = CairoMakie.Axis(fig[2, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_30.grid.xᶜᵃᵃ[field_NN_data_30.indices[1][1]] / 1000) km")
axfield_40 = CairoMakie.Axis(fig[3, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_40.grid.xᶜᵃᵃ[field_NN_data_40.indices[1][1]] / 1000) km")
axfield_50 = CairoMakie.Axis(fig[3, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_50.grid.xᶜᵃᵃ[field_NN_data_50.indices[1][1]] / 1000) km")
axfield_60 = CairoMakie.Axis(fig[4, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_60.grid.xᶜᵃᵃ[field_NN_data_60.indices[1][1]] / 1000) km")
axfield_70 = CairoMakie.Axis(fig[4, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_70.grid.xᶜᵃᵃ[field_NN_data_70.indices[1][1]] / 1000) km")
axfield_80 = CairoMakie.Axis(fig[5, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_80.grid.xᶜᵃᵃ[field_NN_data_80.indices[1][1]] / 1000) km")
axfield_90 = CairoMakie.Axis(fig[5, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_90.grid.xᶜᵃᵃ[field_NN_data_90.indices[1][1]] / 1000) km")

axflux_00 = CairoMakie.Axis(fig[1, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_00.grid.xᶜᵃᵃ[flux_NN_data_00.indices[1][1]] / 1000) km")
axflux_10 = CairoMakie.Axis(fig[1, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_10.grid.xᶜᵃᵃ[flux_NN_data_10.indices[1][1]] / 1000) km")
axflux_20 = CairoMakie.Axis(fig[2, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_20.grid.xᶜᵃᵃ[flux_NN_data_20.indices[1][1]] / 1000) km")
axflux_30 = CairoMakie.Axis(fig[2, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_30.grid.xᶜᵃᵃ[flux_NN_data_30.indices[1][1]] / 1000) km")
axflux_40 = CairoMakie.Axis(fig[3, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_40.grid.xᶜᵃᵃ[flux_NN_data_40.indices[1][1]] / 1000) km")
axflux_50 = CairoMakie.Axis(fig[3, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_50.grid.xᶜᵃᵃ[flux_NN_data_50.indices[1][1]] / 1000) km")
axflux_60 = CairoMakie.Axis(fig[4, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_60.grid.xᶜᵃᵃ[flux_NN_data_60.indices[1][1]] / 1000) km")
axflux_70 = CairoMakie.Axis(fig[4, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_70.grid.xᶜᵃᵃ[flux_NN_data_70.indices[1][1]] / 1000) km")
axflux_80 = CairoMakie.Axis(fig[5, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_80.grid.xᶜᵃᵃ[flux_NN_data_80.indices[1][1]] / 1000) km")
axflux_90 = CairoMakie.Axis(fig[5, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_90.grid.xᶜᵃᵃ[flux_NN_data_90.indices[1][1]] / 1000) km")

n = Observable(1096)

zC_indices = 1:200
zF_indices = 2:200

field_lim = (find_min(interior(field_NN_data_00[timeframes[1]], :, :, zC_indices), interior(field_NN_data_00[timeframes[end]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)),
         find_max(interior(field_NN_data_00[timeframes[1]], :, :, zC_indices), interior(field_NN_data_00[timeframes[end]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)))

flux_lim = (find_min(interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)),
         find_max(interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)))

flux_lim = (-maximum(abs, flux_lim), maximum(abs, flux_lim))

NN_00ₙ = @lift interior(field_NN_data_00[$n], 1, :, zC_indices)
NN_10ₙ = @lift interior(field_NN_data_10[$n], 1, :, zC_indices)
NN_20ₙ = @lift interior(field_NN_data_20[$n], 1, :, zC_indices)
NN_30ₙ = @lift interior(field_NN_data_30[$n], 1, :, zC_indices)
NN_40ₙ = @lift interior(field_NN_data_40[$n], 1, :, zC_indices)
NN_50ₙ = @lift interior(field_NN_data_50[$n], 1, :, zC_indices)
NN_60ₙ = @lift interior(field_NN_data_60[$n], 1, :, zC_indices)
NN_70ₙ = @lift interior(field_NN_data_70[$n], 1, :, zC_indices)
NN_80ₙ = @lift interior(field_NN_data_80[$n], 1, :, zC_indices)
NN_90ₙ = @lift interior(field_NN_data_90[$n], 1, :, zC_indices)

flux_00ₙ = @lift interior(flux_NN_data_00[$n], 1, :, zF_indices)
flux_10ₙ = @lift interior(flux_NN_data_10[$n], 1, :, zF_indices)
flux_20ₙ = @lift interior(flux_NN_data_20[$n], 1, :, zF_indices)
flux_30ₙ = @lift interior(flux_NN_data_30[$n], 1, :, zF_indices)
flux_40ₙ = @lift interior(flux_NN_data_40[$n], 1, :, zF_indices)
flux_50ₙ = @lift interior(flux_NN_data_50[$n], 1, :, zF_indices)
flux_60ₙ = @lift interior(flux_NN_data_60[$n], 1, :, zF_indices)
flux_70ₙ = @lift interior(flux_NN_data_70[$n], 1, :, zF_indices)
flux_80ₙ = @lift interior(flux_NN_data_80[$n], 1, :, zF_indices)
flux_90ₙ = @lift interior(flux_NN_data_90[$n], 1, :, zF_indices)

colorscheme_field = colorschemes[:viridis]
colorscheme_flux = colorschemes[:balance]

field_00_surface = heatmap!(axfield_00, yC, zC[zC_indices], NN_00ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_10_surface = heatmap!(axfield_10, yC, zC[zC_indices], NN_10ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_20_surface = heatmap!(axfield_20, yC, zC[zC_indices], NN_20ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_30_surface = heatmap!(axfield_30, yC, zC[zC_indices], NN_30ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_40_surface = heatmap!(axfield_40, yC, zC[zC_indices], NN_40ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_50_surface = heatmap!(axfield_50, yC, zC[zC_indices], NN_50ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_60_surface = heatmap!(axfield_60, yC, zC[zC_indices], NN_60ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_70_surface = heatmap!(axfield_70, yC, zC[zC_indices], NN_70ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_80_surface = heatmap!(axfield_80, yC, zC[zC_indices], NN_80ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_90_surface = heatmap!(axfield_90, yC, zC[zC_indices], NN_90ₙ, colormap=colorscheme_field, colorrange=field_lim)

flux_00_surface = heatmap!(axflux_00, yC, zC[zF_indices], flux_00ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_10_surface = heatmap!(axflux_10, yC, zC[zF_indices], flux_10ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_20_surface = heatmap!(axflux_20, yC, zC[zF_indices], flux_20ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_30_surface = heatmap!(axflux_30, yC, zC[zF_indices], flux_30ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_40_surface = heatmap!(axflux_40, yC, zC[zF_indices], flux_40ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_50_surface = heatmap!(axflux_50, yC, zC[zF_indices], flux_50ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_60_surface = heatmap!(axflux_60, yC, zC[zF_indices], flux_60ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_70_surface = heatmap!(axflux_70, yC, zC[zF_indices], flux_70ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_80_surface = heatmap!(axflux_80, yC, zC[zF_indices], flux_80ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_90_surface = heatmap!(axflux_90, yC, zC[zF_indices], flux_90ₙ, colormap=colorscheme_flux, colorrange=flux_lim)

Colorbar(fig[1:5, 2], field_00_surface, label="Field")
Colorbar(fig[1:5, 4], flux_00_surface, label="NN Flux")
Colorbar(fig[1:5, 6], field_00_surface, label="Field")
Colorbar(fig[1:5, 8], flux_00_surface, label="NN Flux")

xlims!(axfield_00, minimum(yC), maximum(yC))
xlims!(axfield_10, minimum(yC), maximum(yC))
xlims!(axfield_20, minimum(yC), maximum(yC))
xlims!(axfield_30, minimum(yC), maximum(yC))
xlims!(axfield_40, minimum(yC), maximum(yC))
xlims!(axfield_50, minimum(yC), maximum(yC))
xlims!(axfield_60, minimum(yC), maximum(yC))
xlims!(axfield_70, minimum(yC), maximum(yC))
xlims!(axfield_80, minimum(yC), maximum(yC))
xlims!(axfield_90, minimum(yC), maximum(yC))

ylims!(axfield_00, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_10, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_20, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_30, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_40, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_50, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_60, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_70, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_80, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_90, minimum(zC[zC_indices]), maximum(zC[zC_indices]))

xlims!(axflux_00, minimum(yC), maximum(yC))
xlims!(axflux_10, minimum(yC), maximum(yC))
xlims!(axflux_20, minimum(yC), maximum(yC))
xlims!(axflux_30, minimum(yC), maximum(yC))
xlims!(axflux_40, minimum(yC), maximum(yC))
xlims!(axflux_50, minimum(yC), maximum(yC))
xlims!(axflux_60, minimum(yC), maximum(yC))
xlims!(axflux_70, minimum(yC), maximum(yC))
xlims!(axflux_80, minimum(yC), maximum(yC))
xlims!(axflux_90, minimum(yC), maximum(yC))

ylims!(axflux_00, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_10, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_20, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_30, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_40, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_50, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_60, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_70, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_80, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_90, minimum(zF[zF_indices]), maximum(zF[zF_indices]))

title_str = @lift "Temperature (°C), Time = $(round(times[$n], digits=2)) years"
Label(fig[0, :], text=title_str, tellwidth=false, font=:bold)

trim!(fig.layout)

# save("./Output/compare_3D_instantaneous_fields_slices_NNclosure_fluxes.png", fig)
# display(fig)
CairoMakie.record(fig, "$(FILE_DIR)/$(filename)_yzslices_fluxes_T.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
  n[] = nn
end
#%%
@info "Recording S fields and fluxes in yz"

fieldname = "S"
fluxname = "wS_NN"
field_NN_data_00 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", fieldname, backend=OnDisk())
field_NN_data_10 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_10.jld2", fieldname, backend=OnDisk())
field_NN_data_20 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_20.jld2", fieldname, backend=OnDisk())
field_NN_data_30 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_30.jld2", fieldname, backend=OnDisk())
field_NN_data_40 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_40.jld2", fieldname, backend=OnDisk())
field_NN_data_50 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_50.jld2", fieldname, backend=OnDisk())
field_NN_data_60 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_60.jld2", fieldname, backend=OnDisk())
field_NN_data_70 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_70.jld2", fieldname, backend=OnDisk())
field_NN_data_80 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_80.jld2", fieldname, backend=OnDisk())
field_NN_data_90 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_90.jld2", fieldname, backend=OnDisk())

flux_NN_data_00 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", fluxname, backend=OnDisk())
flux_NN_data_10 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_10.jld2", fluxname, backend=OnDisk())
flux_NN_data_20 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_20.jld2", fluxname, backend=OnDisk())
flux_NN_data_30 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_30.jld2", fluxname, backend=OnDisk())
flux_NN_data_40 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_40.jld2", fluxname, backend=OnDisk())
flux_NN_data_50 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_50.jld2", fluxname, backend=OnDisk())
flux_NN_data_60 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_60.jld2", fluxname, backend=OnDisk())
flux_NN_data_70 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_70.jld2", fluxname, backend=OnDisk())
flux_NN_data_80 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_80.jld2", fluxname, backend=OnDisk())
flux_NN_data_90 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz_90.jld2", fluxname, backend=OnDisk())

xC = field_NN_data_00.grid.xᶜᵃᵃ[1:field_NN_data_00.grid.Nx]
yC = field_NN_data_00.grid.yᵃᶜᵃ[1:field_NN_data_00.grid.Ny]
zC = field_NN_data_00.grid.zᵃᵃᶜ[1:field_NN_data_00.grid.Nz]
zF = field_NN_data_00.grid.zᵃᵃᶠ[1:field_NN_data_00.grid.Nz+1]

Nt = length(field_NN_data_90)
times = field_NN_data_00.times / 24 / 60^2 / 365
timeframes = 1:Nt

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

#%%
fig = Figure(size=(3000, 1200))

axfield_00 = CairoMakie.Axis(fig[1, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_00.grid.xᶜᵃᵃ[field_NN_data_00.indices[1][1]] / 1000) km")
axfield_10 = CairoMakie.Axis(fig[1, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_10.grid.xᶜᵃᵃ[field_NN_data_10.indices[1][1]] / 1000) km")
axfield_20 = CairoMakie.Axis(fig[2, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_20.grid.xᶜᵃᵃ[field_NN_data_20.indices[1][1]] / 1000) km")
axfield_30 = CairoMakie.Axis(fig[2, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_30.grid.xᶜᵃᵃ[field_NN_data_30.indices[1][1]] / 1000) km")
axfield_40 = CairoMakie.Axis(fig[3, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_40.grid.xᶜᵃᵃ[field_NN_data_40.indices[1][1]] / 1000) km")
axfield_50 = CairoMakie.Axis(fig[3, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_50.grid.xᶜᵃᵃ[field_NN_data_50.indices[1][1]] / 1000) km")
axfield_60 = CairoMakie.Axis(fig[4, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_60.grid.xᶜᵃᵃ[field_NN_data_60.indices[1][1]] / 1000) km")
axfield_70 = CairoMakie.Axis(fig[4, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_70.grid.xᶜᵃᵃ[field_NN_data_70.indices[1][1]] / 1000) km")
axfield_80 = CairoMakie.Axis(fig[5, 1], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_80.grid.xᶜᵃᵃ[field_NN_data_80.indices[1][1]] / 1000) km")
axfield_90 = CairoMakie.Axis(fig[5, 5], xlabel="y (m)", ylabel="z (m)", title="x = $(field_NN_data_90.grid.xᶜᵃᵃ[field_NN_data_90.indices[1][1]] / 1000) km")

axflux_00 = CairoMakie.Axis(fig[1, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_00.grid.xᶜᵃᵃ[flux_NN_data_00.indices[1][1]] / 1000) km")
axflux_10 = CairoMakie.Axis(fig[1, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_10.grid.xᶜᵃᵃ[flux_NN_data_10.indices[1][1]] / 1000) km")
axflux_20 = CairoMakie.Axis(fig[2, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_20.grid.xᶜᵃᵃ[flux_NN_data_20.indices[1][1]] / 1000) km")
axflux_30 = CairoMakie.Axis(fig[2, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_30.grid.xᶜᵃᵃ[flux_NN_data_30.indices[1][1]] / 1000) km")
axflux_40 = CairoMakie.Axis(fig[3, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_40.grid.xᶜᵃᵃ[flux_NN_data_40.indices[1][1]] / 1000) km")
axflux_50 = CairoMakie.Axis(fig[3, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_50.grid.xᶜᵃᵃ[flux_NN_data_50.indices[1][1]] / 1000) km")
axflux_60 = CairoMakie.Axis(fig[4, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_60.grid.xᶜᵃᵃ[flux_NN_data_60.indices[1][1]] / 1000) km")
axflux_70 = CairoMakie.Axis(fig[4, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_70.grid.xᶜᵃᵃ[flux_NN_data_70.indices[1][1]] / 1000) km")
axflux_80 = CairoMakie.Axis(fig[5, 3], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_80.grid.xᶜᵃᵃ[flux_NN_data_80.indices[1][1]] / 1000) km")
axflux_90 = CairoMakie.Axis(fig[5, 7], xlabel="y (m)", ylabel="z (m)", title="x = $(flux_NN_data_90.grid.xᶜᵃᵃ[flux_NN_data_90.indices[1][1]] / 1000) km")

n = Observable(1096)

zC_indices = 1:200
zF_indices = 2:200

field_lim = (find_min(interior(field_NN_data_00[timeframes[1]], :, :, zC_indices), interior(field_NN_data_00[timeframes[end]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)),
         find_max(interior(field_NN_data_00[timeframes[1]], :, :, zC_indices), interior(field_NN_data_00[timeframes[end]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)))

flux_lim = (find_min(interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)),
         find_max(interior(flux_NN_data_00[timeframes[1]], :, :, zC_indices), interior(flux_NN_data_00[timeframes[end]], :, :, zC_indices)))

flux_lim = (-maximum(abs, flux_lim), maximum(abs, flux_lim))

NN_00ₙ = @lift interior(field_NN_data_00[$n], 1, :, zC_indices)
NN_10ₙ = @lift interior(field_NN_data_10[$n], 1, :, zC_indices)
NN_20ₙ = @lift interior(field_NN_data_20[$n], 1, :, zC_indices)
NN_30ₙ = @lift interior(field_NN_data_30[$n], 1, :, zC_indices)
NN_40ₙ = @lift interior(field_NN_data_40[$n], 1, :, zC_indices)
NN_50ₙ = @lift interior(field_NN_data_50[$n], 1, :, zC_indices)
NN_60ₙ = @lift interior(field_NN_data_60[$n], 1, :, zC_indices)
NN_70ₙ = @lift interior(field_NN_data_70[$n], 1, :, zC_indices)
NN_80ₙ = @lift interior(field_NN_data_80[$n], 1, :, zC_indices)
NN_90ₙ = @lift interior(field_NN_data_90[$n], 1, :, zC_indices)

flux_00ₙ = @lift interior(flux_NN_data_00[$n], 1, :, zF_indices)
flux_10ₙ = @lift interior(flux_NN_data_10[$n], 1, :, zF_indices)
flux_20ₙ = @lift interior(flux_NN_data_20[$n], 1, :, zF_indices)
flux_30ₙ = @lift interior(flux_NN_data_30[$n], 1, :, zF_indices)
flux_40ₙ = @lift interior(flux_NN_data_40[$n], 1, :, zF_indices)
flux_50ₙ = @lift interior(flux_NN_data_50[$n], 1, :, zF_indices)
flux_60ₙ = @lift interior(flux_NN_data_60[$n], 1, :, zF_indices)
flux_70ₙ = @lift interior(flux_NN_data_70[$n], 1, :, zF_indices)
flux_80ₙ = @lift interior(flux_NN_data_80[$n], 1, :, zF_indices)
flux_90ₙ = @lift interior(flux_NN_data_90[$n], 1, :, zF_indices)

colorscheme_field = colorschemes[:viridis]
colorscheme_flux = colorschemes[:balance]

field_00_surface = heatmap!(axfield_00, yC, zC[zC_indices], NN_00ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_10_surface = heatmap!(axfield_10, yC, zC[zC_indices], NN_10ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_20_surface = heatmap!(axfield_20, yC, zC[zC_indices], NN_20ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_30_surface = heatmap!(axfield_30, yC, zC[zC_indices], NN_30ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_40_surface = heatmap!(axfield_40, yC, zC[zC_indices], NN_40ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_50_surface = heatmap!(axfield_50, yC, zC[zC_indices], NN_50ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_60_surface = heatmap!(axfield_60, yC, zC[zC_indices], NN_60ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_70_surface = heatmap!(axfield_70, yC, zC[zC_indices], NN_70ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_80_surface = heatmap!(axfield_80, yC, zC[zC_indices], NN_80ₙ, colormap=colorscheme_field, colorrange=field_lim)
field_90_surface = heatmap!(axfield_90, yC, zC[zC_indices], NN_90ₙ, colormap=colorscheme_field, colorrange=field_lim)

flux_00_surface = heatmap!(axflux_00, yC, zC[zF_indices], flux_00ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_10_surface = heatmap!(axflux_10, yC, zC[zF_indices], flux_10ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_20_surface = heatmap!(axflux_20, yC, zC[zF_indices], flux_20ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_30_surface = heatmap!(axflux_30, yC, zC[zF_indices], flux_30ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_40_surface = heatmap!(axflux_40, yC, zC[zF_indices], flux_40ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_50_surface = heatmap!(axflux_50, yC, zC[zF_indices], flux_50ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_60_surface = heatmap!(axflux_60, yC, zC[zF_indices], flux_60ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_70_surface = heatmap!(axflux_70, yC, zC[zF_indices], flux_70ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_80_surface = heatmap!(axflux_80, yC, zC[zF_indices], flux_80ₙ, colormap=colorscheme_flux, colorrange=flux_lim)
flux_90_surface = heatmap!(axflux_90, yC, zC[zF_indices], flux_90ₙ, colormap=colorscheme_flux, colorrange=flux_lim)

Colorbar(fig[1:5, 2], field_00_surface, label="Field")
Colorbar(fig[1:5, 4], flux_00_surface, label="NN Flux")
Colorbar(fig[1:5, 6], field_00_surface, label="Field")
Colorbar(fig[1:5, 8], flux_00_surface, label="NN Flux")

xlims!(axfield_00, minimum(yC), maximum(yC))
xlims!(axfield_10, minimum(yC), maximum(yC))
xlims!(axfield_20, minimum(yC), maximum(yC))
xlims!(axfield_30, minimum(yC), maximum(yC))
xlims!(axfield_40, minimum(yC), maximum(yC))
xlims!(axfield_50, minimum(yC), maximum(yC))
xlims!(axfield_60, minimum(yC), maximum(yC))
xlims!(axfield_70, minimum(yC), maximum(yC))
xlims!(axfield_80, minimum(yC), maximum(yC))
xlims!(axfield_90, minimum(yC), maximum(yC))

ylims!(axfield_00, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_10, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_20, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_30, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_40, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_50, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_60, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_70, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_80, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
ylims!(axfield_90, minimum(zC[zC_indices]), maximum(zC[zC_indices]))

xlims!(axflux_00, minimum(yC), maximum(yC))
xlims!(axflux_10, minimum(yC), maximum(yC))
xlims!(axflux_20, minimum(yC), maximum(yC))
xlims!(axflux_30, minimum(yC), maximum(yC))
xlims!(axflux_40, minimum(yC), maximum(yC))
xlims!(axflux_50, minimum(yC), maximum(yC))
xlims!(axflux_60, minimum(yC), maximum(yC))
xlims!(axflux_70, minimum(yC), maximum(yC))
xlims!(axflux_80, minimum(yC), maximum(yC))
xlims!(axflux_90, minimum(yC), maximum(yC))

ylims!(axflux_00, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_10, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_20, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_30, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_40, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_50, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_60, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_70, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_80, minimum(zF[zF_indices]), maximum(zF[zF_indices]))
ylims!(axflux_90, minimum(zF[zF_indices]), maximum(zF[zF_indices]))

title_str = @lift "Salinity (psu), Time = $(round(times[$n], digits=2)) years"
Label(fig[0, :], text=title_str, tellwidth=false, font=:bold)

trim!(fig.layout)

CairoMakie.record(fig, "$(FILE_DIR)/$(filename)_yzslices_fluxes_S.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
  @info "Recording frame $nn"
  n[] = nn
end

#%%
@info "Recording T fields and fluxes in xz"

fieldname = "T"
fluxname = "wT_NN"

field_NN_data_5 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_5.jld2", fieldname, backend=OnDisk())
field_NN_data_15 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_15.jld2", fieldname, backend=OnDisk())
field_NN_data_25 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_25.jld2", fieldname, backend=OnDisk())
field_NN_data_35 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_35.jld2", fieldname, backend=OnDisk())
field_NN_data_45 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_45.jld2", fieldname, backend=OnDisk())
field_NN_data_55 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_55.jld2", fieldname, backend=OnDisk())
field_NN_data_65 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_65.jld2", fieldname, backend=OnDisk())
field_NN_data_75 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_75.jld2", fieldname, backend=OnDisk())
field_NN_data_85 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_85.jld2", fieldname, backend=OnDisk())
field_NN_data_95 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_95.jld2", fieldname, backend=OnDisk())

flux_NN_data_5 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_5.jld2", fluxname, backend=OnDisk())
flux_NN_data_15 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_15.jld2", fluxname, backend=OnDisk())
flux_NN_data_25 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_25.jld2", fluxname, backend=OnDisk())
flux_NN_data_35 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_35.jld2", fluxname, backend=OnDisk())
flux_NN_data_45 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_45.jld2", fluxname, backend=OnDisk())
flux_NN_data_55 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_55.jld2", fluxname, backend=OnDisk())
flux_NN_data_65 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_65.jld2", fluxname, backend=OnDisk())
flux_NN_data_75 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_75.jld2", fluxname, backend=OnDisk())
flux_NN_data_85 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_85.jld2", fluxname, backend=OnDisk())
flux_NN_data_95 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_95.jld2", fluxname, backend=OnDisk())

field_datas = [field_NN_data_5, field_NN_data_15, field_NN_data_25, field_NN_data_35, field_NN_data_45, field_NN_data_55, field_NN_data_65, field_NN_data_75, field_NN_data_85, field_NN_data_95]
flux_datas = [flux_NN_data_5, flux_NN_data_15, flux_NN_data_25, flux_NN_data_35, flux_NN_data_45, flux_NN_data_55, flux_NN_data_65, flux_NN_data_75, flux_NN_data_85, flux_NN_data_95]

xC = field_NN_data_5.grid.xᶜᵃᵃ[1:field_NN_data_5.grid.Nx]
yC = field_NN_data_5.grid.yᵃᶜᵃ[1:field_NN_data_5.grid.Ny]
zC = field_NN_data_5.grid.zᵃᵃᶜ[1:field_NN_data_5.grid.Nz]
zF = field_NN_data_5.grid.zᵃᵃᶠ[1:field_NN_data_5.grid.Nz+1]

Nt = length(field_NN_data_95)
times = field_NN_data_5.times / 24 / 60^2 / 365
timeframes = 1:Nt

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

#%%
fig = Figure(size=(3000, 1200))

axfield_5 = CairoMakie.Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_5.grid.yᵃᶜᵃ[field_NN_data_5.indices[2][1]] / 1000) km")
axfield_15 = CairoMakie.Axis(fig[1, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_15.grid.yᵃᶜᵃ[field_NN_data_15.indices[2][1]] / 1000) km")
axfield_25 = CairoMakie.Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_25.grid.yᵃᶜᵃ[field_NN_data_25.indices[2][1]] / 1000) km")
axfield_35 = CairoMakie.Axis(fig[2, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_35.grid.yᵃᶜᵃ[field_NN_data_35.indices[2][1]] / 1000) km")
axfield_45 = CairoMakie.Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_45.grid.yᵃᶜᵃ[field_NN_data_45.indices[2][1]] / 1000) km")
axfield_55 = CairoMakie.Axis(fig[3, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_55.grid.yᵃᶜᵃ[field_NN_data_55.indices[2][1]] / 1000) km")
axfield_65 = CairoMakie.Axis(fig[4, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_65.grid.yᵃᶜᵃ[field_NN_data_65.indices[2][1]] / 1000) km")
axfield_75 = CairoMakie.Axis(fig[4, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_75.grid.yᵃᶜᵃ[field_NN_data_75.indices[2][1]] / 1000) km")
axfield_85 = CairoMakie.Axis(fig[5, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_85.grid.yᵃᶜᵃ[field_NN_data_85.indices[2][1]] / 1000) km")
axfield_95 = CairoMakie.Axis(fig[5, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_95.grid.yᵃᶜᵃ[field_NN_data_95.indices[2][1]] / 1000) km")

axflux_5 = CairoMakie.Axis(fig[1, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_5.grid.yᵃᶜᵃ[flux_NN_data_5.indices[2][1]] / 1000) km")
axflux_15 = CairoMakie.Axis(fig[1, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_15.grid.yᵃᶜᵃ[flux_NN_data_15.indices[2][1]] / 1000) km")
axflux_25 = CairoMakie.Axis(fig[2, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_25.grid.yᵃᶜᵃ[flux_NN_data_25.indices[2][1]] / 1000) km")
axflux_35 = CairoMakie.Axis(fig[2, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_35.grid.yᵃᶜᵃ[flux_NN_data_35.indices[2][1]] / 1000) km")
axflux_45 = CairoMakie.Axis(fig[3, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_45.grid.yᵃᶜᵃ[flux_NN_data_45.indices[2][1]] / 1000) km")
axflux_55 = CairoMakie.Axis(fig[3, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_55.grid.yᵃᶜᵃ[flux_NN_data_55.indices[2][1]] / 1000) km")
axflux_65 = CairoMakie.Axis(fig[4, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_65.grid.yᵃᶜᵃ[flux_NN_data_65.indices[2][1]] / 1000) km")
axflux_75 = CairoMakie.Axis(fig[4, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_75.grid.yᵃᶜᵃ[flux_NN_data_75.indices[2][1]] / 1000) km")
axflux_85 = CairoMakie.Axis(fig[5, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_85.grid.yᵃᶜᵃ[flux_NN_data_85.indices[2][1]] / 1000) km")
axflux_95 = CairoMakie.Axis(fig[5, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_95.grid.yᵃᶜᵃ[flux_NN_data_95.indices[2][1]] / 1000) km")

axfields = [axfield_5, axfield_15, axfield_25, axfield_35, axfield_45, axfield_55, axfield_65, axfield_75, axfield_85, axfield_95]
axfluxes = [axflux_5, axflux_15, axflux_25, axflux_35, axflux_45, axflux_55, axflux_65, axflux_75, axflux_85, axflux_95]

n = Observable(1096)

zC_indices = 1:200
zF_indices = 2:200

field_lims = [(find_min(interior(field_data[timeframes[1]], :, :, zC_indices), interior(field_data[timeframes[end]], :, :, zC_indices)), 
               find_max(interior(field_data[timeframes[1]], :, :, zC_indices), interior(field_data[timeframes[end]], :, :, zC_indices))) for field_data in field_datas]

flux_lims = [(find_min(interior(flux_data[timeframes[1]], :, :, zC_indices), interior(flux_data[timeframes[end]], :, :, zC_indices)),
              find_max(interior(flux_data[timeframes[1]], :, :, zC_indices), interior(flux_data[timeframes[end]], :, :, zC_indices))) for flux_data in flux_datas]

flux_lims = [(-maximum(abs, flux_lim), maximum(abs, flux_lim)) for flux_lim in flux_lims]

NNₙs = [@lift interior(field_data[$n], :, 1, zC_indices) for field_data in field_datas]
fluxₙs = [@lift interior(flux_data[$n], :, 1, zF_indices) for flux_data in flux_datas]

colorscheme_field = colorschemes[:viridis]
colorscheme_flux = colorschemes[:balance]

field_5_surface = heatmap!(axfield_5, xC, zC[zC_indices], NN_5ₙ, colormap=colorscheme_field, colorrange=field_lims[1])
field_15_surface = heatmap!(axfield_15, xC, zC[zC_indices], NN_15ₙ, colormap=colorscheme_field, colorrange=field_lims[2])
field_25_surface = heatmap!(axfield_25, xC, zC[zC_indices], NN_25ₙ, colormap=colorscheme_field, colorrange=field_lims[3])
field_35_surface = heatmap!(axfield_35, xC, zC[zC_indices], NN_35ₙ, colormap=colorscheme_field, colorrange=field_lims[4])
field_45_surface = heatmap!(axfield_45, xC, zC[zC_indices], NN_45ₙ, colormap=colorscheme_field, colorrange=field_lims[5])
field_55_surface = heatmap!(axfield_55, xC, zC[zC_indices], NN_55ₙ, colormap=colorscheme_field, colorrange=field_lims[6])
field_65_surface = heatmap!(axfield_65, xC, zC[zC_indices], NN_65ₙ, colormap=colorscheme_field, colorrange=field_lims[7])
field_75_surface = heatmap!(axfield_75, xC, zC[zC_indices], NN_75ₙ, colormap=colorscheme_field, colorrange=field_lims[8])
field_85_surface = heatmap!(axfield_85, xC, zC[zC_indices], NN_85ₙ, colormap=colorscheme_field, colorrange=field_lims[9])
field_95_surface = heatmap!(axfield_95, xC, zC[zC_indices], NN_95ₙ, colormap=colorscheme_field, colorrange=field_lims[10])

flux_5_surface = heatmap!(axflux_5, xC, zC[zF_indices], flux_5ₙ, colormap=colorscheme_flux, colorrange=flux_lims[1])
flux_15_surface = heatmap!(axflux_15, xC, zC[zF_indices], flux_15ₙ, colormap=colorscheme_flux, colorrange=flux_lims[2])
flux_25_surface = heatmap!(axflux_25, xC, zC[zF_indices], flux_25ₙ, colormap=colorscheme_flux, colorrange=flux_lims[3])
flux_35_surface = heatmap!(axflux_35, xC, zC[zF_indices], flux_35ₙ, colormap=colorscheme_flux, colorrange=flux_lims[4])
flux_45_surface = heatmap!(axflux_45, xC, zC[zF_indices], flux_45ₙ, colormap=colorscheme_flux, colorrange=flux_lims[5])
flux_55_surface = heatmap!(axflux_55, xC, zC[zF_indices], flux_55ₙ, colormap=colorscheme_flux, colorrange=flux_lims[6])
flux_65_surface = heatmap!(axflux_65, xC, zC[zF_indices], flux_65ₙ, colormap=colorscheme_flux, colorrange=flux_lims[7])
flux_75_surface = heatmap!(axflux_75, xC, zC[zF_indices], flux_75ₙ, colormap=colorscheme_flux, colorrange=flux_lims[8])
flux_85_surface = heatmap!(axflux_85, xC, zC[zF_indices], flux_85ₙ, colormap=colorscheme_flux, colorrange=flux_lims[9])
flux_95_surface = heatmap!(axflux_95, xC, zC[zF_indices], flux_95ₙ, colormap=colorscheme_flux, colorrange=flux_lims[10])

Colorbar(fig[1, 2], field_5_surface, label="Field")
Colorbar(fig[1, 4], flux_5_surface, label="NN Flux")
Colorbar(fig[1, 6], field_15_surface, label="Field")
Colorbar(fig[1, 8], flux_15_surface, label="NN Flux")
Colorbar(fig[2, 2], field_25_surface, label="Field")
Colorbar(fig[2, 4], flux_25_surface, label="NN Flux")
Colorbar(fig[2, 6], field_35_surface, label="Field")
Colorbar(fig[2, 8], flux_35_surface, label="NN Flux")
Colorbar(fig[3, 2], field_45_surface, label="Field")
Colorbar(fig[3, 4], flux_45_surface, label="NN Flux")
Colorbar(fig[3, 6], field_55_surface, label="Field")
Colorbar(fig[3, 8], flux_55_surface, label="NN Flux")
Colorbar(fig[4, 2], field_65_surface, label="Field")
Colorbar(fig[4, 4], flux_65_surface, label="NN Flux")
Colorbar(fig[4, 6], field_75_surface, label="Field")
Colorbar(fig[4, 8], flux_75_surface, label="NN Flux")
Colorbar(fig[5, 2], field_85_surface, label="Field")
Colorbar(fig[5, 4], flux_85_surface, label="NN Flux")
Colorbar(fig[5, 6], field_95_surface, label="Field")
Colorbar(fig[5, 8], flux_95_surface, label="NN Flux")

for axfield in axfields
  xlims!(axfield, minimum(xC), maximum(xC))
  ylims!(axfield, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
end

for axflux in axfluxes
  xlims!(axflux, minimum(xC), maximum(xC))
  ylims!(axflux, minimum(zC[zF_indices]), maximum(zC[zF_indices]))
end

title_str = @lift "Temperature (°C), Time = $(round(times[$n], digits=2)) years"
Label(fig[0, :], text=title_str, tellwidth=false, font=:bold)

trim!(fig.layout)

CairoMakie.record(fig, "$(FILE_DIR)/$(filename)_xzslices_fluxes_T.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
  n[] = nn
end

#%%
@info "Recording S fields and fluxes in xz"

fieldname = "S"
fluxname = "wS_NN"

field_NN_data_5 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_5.jld2", fieldname, backend=OnDisk())
field_NN_data_15 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_15.jld2", fieldname, backend=OnDisk())
field_NN_data_25 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_25.jld2", fieldname, backend=OnDisk())
field_NN_data_35 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_35.jld2", fieldname, backend=OnDisk())
field_NN_data_45 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_45.jld2", fieldname, backend=OnDisk())
field_NN_data_55 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_55.jld2", fieldname, backend=OnDisk())
field_NN_data_65 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_65.jld2", fieldname, backend=OnDisk())
field_NN_data_75 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_75.jld2", fieldname, backend=OnDisk())
field_NN_data_85 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_85.jld2", fieldname, backend=OnDisk())
field_NN_data_95 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_95.jld2", fieldname, backend=OnDisk())

flux_NN_data_5 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_5.jld2", fluxname, backend=OnDisk())
flux_NN_data_15 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_15.jld2", fluxname, backend=OnDisk())
flux_NN_data_25 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_25.jld2", fluxname, backend=OnDisk())
flux_NN_data_35 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_35.jld2", fluxname, backend=OnDisk())
flux_NN_data_45 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_45.jld2", fluxname, backend=OnDisk())
flux_NN_data_55 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_55.jld2", fluxname, backend=OnDisk())
flux_NN_data_65 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_65.jld2", fluxname, backend=OnDisk())
flux_NN_data_75 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_75.jld2", fluxname, backend=OnDisk())
flux_NN_data_85 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_85.jld2", fluxname, backend=OnDisk())
flux_NN_data_95 = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz_95.jld2", fluxname, backend=OnDisk())

field_datas = [field_NN_data_5, field_NN_data_15, field_NN_data_25, field_NN_data_35, field_NN_data_45, field_NN_data_55, field_NN_data_65, field_NN_data_75, field_NN_data_85, field_NN_data_95]
flux_datas = [flux_NN_data_5, flux_NN_data_15, flux_NN_data_25, flux_NN_data_35, flux_NN_data_45, flux_NN_data_55, flux_NN_data_65, flux_NN_data_75, flux_NN_data_85, flux_NN_data_95]

xC = field_NN_data_5.grid.xᶜᵃᵃ[1:field_NN_data_5.grid.Nx]
yC = field_NN_data_5.grid.yᵃᶜᵃ[1:field_NN_data_5.grid.Ny]
zC = field_NN_data_5.grid.zᵃᵃᶜ[1:field_NN_data_5.grid.Nz]
zF = field_NN_data_5.grid.zᵃᵃᶠ[1:field_NN_data_5.grid.Nz+1]

Nt = length(field_NN_data_95)
times = field_NN_data_5.times / 24 / 60^2 / 365
timeframes = 1:Nt

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

#%%
fig = Figure(size=(3000, 1200))

axfield_5 = CairoMakie.Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_5.grid.yᵃᶜᵃ[field_NN_data_5.indices[2][1]] / 1000) km")
axfield_15 = CairoMakie.Axis(fig[1, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_15.grid.yᵃᶜᵃ[field_NN_data_15.indices[2][1]] / 1000) km")
axfield_25 = CairoMakie.Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_25.grid.yᵃᶜᵃ[field_NN_data_25.indices[2][1]] / 1000) km")
axfield_35 = CairoMakie.Axis(fig[2, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_35.grid.yᵃᶜᵃ[field_NN_data_35.indices[2][1]] / 1000) km")
axfield_45 = CairoMakie.Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_45.grid.yᵃᶜᵃ[field_NN_data_45.indices[2][1]] / 1000) km")
axfield_55 = CairoMakie.Axis(fig[3, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_55.grid.yᵃᶜᵃ[field_NN_data_55.indices[2][1]] / 1000) km")
axfield_65 = CairoMakie.Axis(fig[4, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_65.grid.yᵃᶜᵃ[field_NN_data_65.indices[2][1]] / 1000) km")
axfield_75 = CairoMakie.Axis(fig[4, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_75.grid.yᵃᶜᵃ[field_NN_data_75.indices[2][1]] / 1000) km")
axfield_85 = CairoMakie.Axis(fig[5, 1], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_85.grid.yᵃᶜᵃ[field_NN_data_85.indices[2][1]] / 1000) km")
axfield_95 = CairoMakie.Axis(fig[5, 5], xlabel="x (m)", ylabel="z (m)", title="y = $(field_NN_data_95.grid.yᵃᶜᵃ[field_NN_data_95.indices[2][1]] / 1000) km")

axflux_5 = CairoMakie.Axis(fig[1, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_5.grid.yᵃᶜᵃ[flux_NN_data_5.indices[2][1]] / 1000) km")
axflux_15 = CairoMakie.Axis(fig[1, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_15.grid.yᵃᶜᵃ[flux_NN_data_15.indices[2][1]] / 1000) km")
axflux_25 = CairoMakie.Axis(fig[2, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_25.grid.yᵃᶜᵃ[flux_NN_data_25.indices[2][1]] / 1000) km")
axflux_35 = CairoMakie.Axis(fig[2, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_35.grid.yᵃᶜᵃ[flux_NN_data_35.indices[2][1]] / 1000) km")
axflux_45 = CairoMakie.Axis(fig[3, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_45.grid.yᵃᶜᵃ[flux_NN_data_45.indices[2][1]] / 1000) km")
axflux_55 = CairoMakie.Axis(fig[3, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_55.grid.yᵃᶜᵃ[flux_NN_data_55.indices[2][1]] / 1000) km")
axflux_65 = CairoMakie.Axis(fig[4, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_65.grid.yᵃᶜᵃ[flux_NN_data_65.indices[2][1]] / 1000) km")
axflux_75 = CairoMakie.Axis(fig[4, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_75.grid.yᵃᶜᵃ[flux_NN_data_75.indices[2][1]] / 1000) km")
axflux_85 = CairoMakie.Axis(fig[5, 3], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_85.grid.yᵃᶜᵃ[flux_NN_data_85.indices[2][1]] / 1000) km")
axflux_95 = CairoMakie.Axis(fig[5, 7], xlabel="x (m)", ylabel="z (m)", title="y = $(flux_NN_data_95.grid.yᵃᶜᵃ[flux_NN_data_95.indices[2][1]] / 1000) km")

axfields = [axfield_5, axfield_15, axfield_25, axfield_35, axfield_45, axfield_55, axfield_65, axfield_75, axfield_85, axfield_95]
axfluxes = [axflux_5, axflux_15, axflux_25, axflux_35, axflux_45, axflux_55, axflux_65, axflux_75, axflux_85, axflux_95]

n = Observable(1096)

zC_indices = 1:200
zF_indices = 2:200

field_lims = [(find_min(interior(field_data[timeframes[1]], :, :, zC_indices), interior(field_data[timeframes[end]], :, :, zC_indices)), 
               find_max(interior(field_data[timeframes[1]], :, :, zC_indices), interior(field_data[timeframes[end]], :, :, zC_indices))) for field_data in field_datas]

flux_lims = [(find_min(interior(flux_data[timeframes[1]], :, :, zC_indices), interior(flux_data[timeframes[end]], :, :, zC_indices)),
              find_max(interior(flux_data[timeframes[1]], :, :, zC_indices), interior(flux_data[timeframes[end]], :, :, zC_indices))) for flux_data in flux_datas]

flux_lims = [(-maximum(abs, flux_lim), maximum(abs, flux_lim)) for flux_lim in flux_lims]

NNₙs = [@lift interior(field_data[$n], :, 1, zC_indices) for field_data in field_datas]
fluxₙs = [@lift interior(flux_data[$n], :, 1, zF_indices) for flux_data in flux_datas]

colorscheme_field = colorschemes[:viridis]
colorscheme_flux = colorschemes[:balance]

field_5_surface = heatmap!(axfield_5, xC, zC[zC_indices], NN_5ₙ, colormap=colorscheme_field, colorrange=field_lims[1])
field_15_surface = heatmap!(axfield_15, xC, zC[zC_indices], NN_15ₙ, colormap=colorscheme_field, colorrange=field_lims[2])
field_25_surface = heatmap!(axfield_25, xC, zC[zC_indices], NN_25ₙ, colormap=colorscheme_field, colorrange=field_lims[3])
field_35_surface = heatmap!(axfield_35, xC, zC[zC_indices], NN_35ₙ, colormap=colorscheme_field, colorrange=field_lims[4])
field_45_surface = heatmap!(axfield_45, xC, zC[zC_indices], NN_45ₙ, colormap=colorscheme_field, colorrange=field_lims[5])
field_55_surface = heatmap!(axfield_55, xC, zC[zC_indices], NN_55ₙ, colormap=colorscheme_field, colorrange=field_lims[6])
field_65_surface = heatmap!(axfield_65, xC, zC[zC_indices], NN_65ₙ, colormap=colorscheme_field, colorrange=field_lims[7])
field_75_surface = heatmap!(axfield_75, xC, zC[zC_indices], NN_75ₙ, colormap=colorscheme_field, colorrange=field_lims[8])
field_85_surface = heatmap!(axfield_85, xC, zC[zC_indices], NN_85ₙ, colormap=colorscheme_field, colorrange=field_lims[9])
field_95_surface = heatmap!(axfield_95, xC, zC[zC_indices], NN_95ₙ, colormap=colorscheme_field, colorrange=field_lims[10])

flux_5_surface = heatmap!(axflux_5, xC, zC[zF_indices], flux_5ₙ, colormap=colorscheme_flux, colorrange=flux_lims[1])
flux_15_surface = heatmap!(axflux_15, xC, zC[zF_indices], flux_15ₙ, colormap=colorscheme_flux, colorrange=flux_lims[2])
flux_25_surface = heatmap!(axflux_25, xC, zC[zF_indices], flux_25ₙ, colormap=colorscheme_flux, colorrange=flux_lims[3])
flux_35_surface = heatmap!(axflux_35, xC, zC[zF_indices], flux_35ₙ, colormap=colorscheme_flux, colorrange=flux_lims[4])
flux_45_surface = heatmap!(axflux_45, xC, zC[zF_indices], flux_45ₙ, colormap=colorscheme_flux, colorrange=flux_lims[5])
flux_55_surface = heatmap!(axflux_55, xC, zC[zF_indices], flux_55ₙ, colormap=colorscheme_flux, colorrange=flux_lims[6])
flux_65_surface = heatmap!(axflux_65, xC, zC[zF_indices], flux_65ₙ, colormap=colorscheme_flux, colorrange=flux_lims[7])
flux_75_surface = heatmap!(axflux_75, xC, zC[zF_indices], flux_75ₙ, colormap=colorscheme_flux, colorrange=flux_lims[8])
flux_85_surface = heatmap!(axflux_85, xC, zC[zF_indices], flux_85ₙ, colormap=colorscheme_flux, colorrange=flux_lims[9])
flux_95_surface = heatmap!(axflux_95, xC, zC[zF_indices], flux_95ₙ, colormap=colorscheme_flux, colorrange=flux_lims[10])

Colorbar(fig[1, 2], field_5_surface, label="Field")
Colorbar(fig[1, 4], flux_5_surface, label="NN Flux")
Colorbar(fig[1, 6], field_15_surface, label="Field")
Colorbar(fig[1, 8], flux_15_surface, label="NN Flux")
Colorbar(fig[2, 2], field_25_surface, label="Field")
Colorbar(fig[2, 4], flux_25_surface, label="NN Flux")
Colorbar(fig[2, 6], field_35_surface, label="Field")
Colorbar(fig[2, 8], flux_35_surface, label="NN Flux")
Colorbar(fig[3, 2], field_45_surface, label="Field")
Colorbar(fig[3, 4], flux_45_surface, label="NN Flux")
Colorbar(fig[3, 6], field_55_surface, label="Field")
Colorbar(fig[3, 8], flux_55_surface, label="NN Flux")
Colorbar(fig[4, 2], field_65_surface, label="Field")
Colorbar(fig[4, 4], flux_65_surface, label="NN Flux")
Colorbar(fig[4, 6], field_75_surface, label="Field")
Colorbar(fig[4, 8], flux_75_surface, label="NN Flux")
Colorbar(fig[5, 2], field_85_surface, label="Field")
Colorbar(fig[5, 4], flux_85_surface, label="NN Flux")
Colorbar(fig[5, 6], field_95_surface, label="Field")
Colorbar(fig[5, 8], flux_95_surface, label="NN Flux")

for axfield in axfields
  xlims!(axfield, minimum(xC), maximum(xC))
  ylims!(axfield, minimum(zC[zC_indices]), maximum(zC[zC_indices]))
end

for axflux in axfluxes
  xlims!(axflux, minimum(xC), maximum(xC))
  ylims!(axflux, minimum(zC[zF_indices]), maximum(zC[zF_indices]))
end

title_str = @lift "Salinity (psu), Time = $(round(times[$n], digits=2)) years"
Label(fig[0, :], text=title_str, tellwidth=false, font=:bold)

trim!(fig.layout)

CairoMakie.record(fig, "$(FILE_DIR)/$(filename)_xzslices_fluxes_S.mp4", 1:Nt, framerate=15, px_per_unit=2) do nn
  n[] = nn
end
#%%