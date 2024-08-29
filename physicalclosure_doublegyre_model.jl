#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
# include("NN_closure_global.jl")
# include("xin_kai_vertical_diffusivity_local.jl")
# include("xin_kai_vertical_diffusivity_2Pr.jl")

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
import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ
s(Sᴬ::Number) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN

#%%
# Architecture
model_architecture = GPU()

# nn_closure = NNFluxClosure(model_architecture)
# base_closure = XinKaiLocalVerticalDiffusivity()
# closure = (nn_closure, base_closure)

vertical_base_closure = VerticalScalarDiffusivity(ν=1e-5, κ=1e-5)
# convection_closure = XinKaiVerticalDiffusivity()
convection_closure = RiBasedVerticalDiffusivity()
closure = (vertical_base_closure, convection_closure)
# closure = vertical_base_closure

# number of grid points
const Nx = 96
const Ny = 96
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

@inline T_initial(x, y, z) = T_north + ΔT / 2 * (1 + z / Lz)

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

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
    momentum_advection = WENO(order=5),
    tracer_advection = WENO(order=5),
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    # closure = (nn_closure, base_closure),
    closure = closure,
    # closure = RiBasedVerticalDiffusivity(),
    tracers = (:T, :S),
    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
)

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
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
stop_time = 730days

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
# ν, κ = model.diffusivity_fields[2].κᵘ, model.diffusivity_fields[2].κᶜ
# Ri = model.diffusivity_fields[2].Ri
# wT, wS = model.diffusivity_fields[2].wT, model.diffusivity_fields[2].wS

# outputs = (; u, v, w, T, S, ν, κ, Ri, wT, wS)
# outputs = (; u, v, w, T, S, ν, κ, Ri)
outputs = (; u, v, w, T, S)

#####
##### Build checkpointer and output writer
#####
simulation.output_writers[:xy] = JLD2OutputWriter(model, outputs,
                                                    # filename = "NN_closure_2D_channel_NDE_FC_Qb_18simnew_2layer_128_relu_2Pr",
                                                    filename = "doublegyre_Ri_based_vertical_diffusivity_2Pr_xy",
                                                    indices = (:, :, Nz),
                                                    schedule = TimeInterval(1day),
                                                    overwrite_existing = true)

simulation.output_writers[:yz] = JLD2OutputWriter(model, outputs,
                                                    # filename = "NN_closure_2D_channel_NDE_FC_Qb_18simnew_2layer_128_relu_2Pr",
                                                    filename = "doublegyre_Ri_based_vertical_diffusivity_2Pr_yz",
                                                    indices = (1, :, :),
                                                    schedule = TimeInterval(1day),
                                                    overwrite_existing = true)
                                                    
simulation.output_writers[:xz] = JLD2OutputWriter(model, outputs,
                                                    # filename = "NN_closure_2D_channel_NDE_FC_Qb_18simnew_2layer_128_relu_2Pr",
                                                    filename = "doublegyre_Ri_based_vertical_diffusivity_2Pr_xz",
                                                    indices = (:, 1, :),
                                                    schedule = TimeInterval(1day),
                                                    overwrite_existing = true)

@info "Running the simulation..."

try
    run!(simulation, pickup = false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end
#%%
T_xy_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xy.jld2", "T")
T_xz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xz.jld2", "T")
T_yz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_yz.jld2", "T")

S_xy_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xy.jld2", "S")
S_xz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xz.jld2", "S")
S_yz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_yz.jld2", "S")

u_xy_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xy.jld2", "u")
u_xz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xz.jld2", "u")
u_yz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_yz.jld2", "u")

v_xy_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xy.jld2", "v")
v_xz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_xz.jld2", "v")
v_yz_data = FieldTimeSeries("doublegyre_Ri_based_vertical_diffusivity_2Pr_yz.jld2", "v")

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
axT = Axis3(fig[1, 1], title="Temperature (°C)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axS = Axis3(fig[1, 3], title="Salinity (g kg⁻¹)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axu = Axis3(fig[2, 1], title="u (m/s)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axv = Axis3(fig[2, 3], title="v (m/s)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)

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

CairoMakie.record(fig, "./doublegyre_Ri_based_vertical_diffusivity_2Pr.mp4", 1:Nt, framerate=20, px_per_unit=2) do nn
    @info nn
    n[] = nn
end

# display(fig)
#%%

# # #####
# # ##### Visualization
# # #####
# using CairoMakie

# dataname = "NN_closure_doublegyre_NDE_FC_Qb_absf_24simnew_2layer_128_relu_2Pr"
# DATA_DIR = "./$(dataname).jld2"

# u_data = FieldTimeSeries("$(DATA_DIR)", "u")
# v_data = FieldTimeSeries("$(DATA_DIR)", "v")
# T_data = FieldTimeSeries("$(DATA_DIR)", "T")
# S_data = FieldTimeSeries("$(DATA_DIR)", "S")
# # ν_data = FieldTimeSeries("$(DATA_DIR)", "ν")
# # κ_data = FieldTimeSeries("$(DATA_DIR)", "κ")
# # Ri_data = FieldTimeSeries("$(DATA_DIR)", "Ri")
# # wT_data = FieldTimeSeries("$(DATA_DIR)", "wT")
# # wS_data = FieldTimeSeries("$(DATA_DIR)", "wS")

# yC = ynodes(T_data.grid, Center())
# yF = ynodes(T_data.grid, Face())

# zC = znodes(T_data.grid, Center())
# zF = znodes(T_data.grid, Face())

# Nt = length(T_data.times)
# #%%
# fig = Figure(size = (1500, 900))
# axu = CairoMakie.Axis(fig[1, 1], xlabel = "y (m)", ylabel = "z (m)", title = "u (m/s)")
# axv = CairoMakie.Axis(fig[1, 3], xlabel = "y (m)", ylabel = "z (m)", title = "v (m/s)")
# axT = CairoMakie.Axis(fig[2, 1], xlabel = "y (m)", ylabel = "z (m)", title = "Temperature (°C)")
# axS = CairoMakie.Axis(fig[2, 3], xlabel = "y (m)", ylabel = "z (m)", title = "Salinity (psu)")
# n = Observable(1)

# uₙ = @lift interior(u_data[$n], 45, :, :)
# vₙ = @lift interior(v_data[$n], 45, :, :)
# Tₙ = @lift interior(T_data[$n], 45, :, :)
# Sₙ = @lift interior(S_data[$n], 45, :, :)

# ulim = @lift (-maximum([maximum(abs, $uₙ), 1e-16]), maximum([maximum(abs, $uₙ),  1e-16]))
# vlim = @lift (-maximum([maximum(abs, $vₙ), 1e-16]), maximum([maximum(abs, $vₙ), 1e-16]))
# Tlim = (minimum(interior(T_data[1])), maximum(interior(T_data[1])))
# Slim = (minimum(interior(S_data[1])), maximum(interior(S_data[1])))

# title_str = @lift "Time: $(round(T_data.times[$n] / 86400, digits=2)) days"
# Label(fig[0, :], title_str, tellwidth = false)

# hu = heatmap!(axu, yC, zC, uₙ, colormap=:RdBu_9, colorrange=ulim)
# hv = heatmap!(axv, yF, zC, vₙ, colormap=:RdBu_9, colorrange=vlim)
# hT = heatmap!(axT, yC, zC, Tₙ, colorrange=Tlim)
# hS = heatmap!(axS, yC, zC, Sₙ, colorrange=Slim)

# Colorbar(fig[1, 2], hu, label = "u (m/s)")
# Colorbar(fig[1, 4], hv, label = "v (m/s)")
# Colorbar(fig[2, 2], hT, label = "T (°C)")
# Colorbar(fig[2, 4], hS, label = "S (psu)")

# CairoMakie.record(fig, "./$(dataname)_test.mp4", 1:Nt, framerate=10) do nn
#     n[] = nn
# end

# display(fig)
# #%%
# fig = Figure(size = (1920, 1080))
# axu = CairoMakie.Axis(fig[1, 1], xlabel = "y (m)", ylabel = "z (m)", title = "u")
# axv = CairoMakie.Axis(fig[1, 3], xlabel = "y (m)", ylabel = "z (m)", title = "v")
# axT = CairoMakie.Axis(fig[2, 1], xlabel = "y (m)", ylabel = "z (m)", title = "Temperature")
# axS = CairoMakie.Axis(fig[2, 3], xlabel = "y (m)", ylabel = "z (m)", title = "Salinity")
# axν = CairoMakie.Axis(fig[1, 5], xlabel = "y (m)", ylabel = "z (m)", title = "Viscosity (log10 scale)")
# axκ = CairoMakie.Axis(fig[2, 5], xlabel = "y (m)", ylabel = "z (m)", title = "Diffusivity (log10 scale)")
# axRi = CairoMakie.Axis(fig[3, 5], xlabel = "y (m)", ylabel = "z (m)", title = "Richardson number")
# axwT = CairoMakie.Axis(fig[3, 1], xlabel = "y (m)", ylabel = "z (m)", title = "wT(NN)")
# axwS = CairoMakie.Axis(fig[3, 3], xlabel = "y (m)", ylabel = "z (m)", title = "wS(NN)")

# n = Observable(1)

# uₙ = @lift interior(u_data[$n], 1, :, :)
# vₙ = @lift interior(v_data[$n], 1, :, :)
# Tₙ = @lift interior(T_data[$n], 1, :, :)
# Sₙ = @lift interior(S_data[$n], 1, :, :)
# νₙ = @lift log10.(interior(ν_data[$n], 1, :, :))
# κₙ = @lift log10.(interior(κ_data[$n], 1, :, :))
# Riₙ = @lift clamp.(interior(Ri_data[$n], 1, :, :), -20, 20)
# wTₙ = @lift interior(wT_data[$n], 1, :, :)
# wSₙ = @lift interior(wS_data[$n], 1, :, :)

# ulim = @lift (-maximum([maximum(abs, $uₙ), 1e-7]), maximum([maximum(abs, $uₙ), 1e-7]))
# vlim = @lift (-maximum([maximum(abs, $vₙ), 1e-7]), maximum([maximum(abs, $vₙ), 1e-7]))
# Tlim = (minimum(interior(T_data[1])), maximum(interior(T_data[1])))
# Slim = (minimum(interior(S_data[1])), maximum(interior(S_data[1])))
# νlim = (-6, 2)
# κlim = (-6, 2)
# wTlim = @lift (-maximum([maximum(abs, $wTₙ), 1e-7]), maximum([maximum(abs, $wTₙ), 1e-7]))
# wSlim = @lift (-maximum([maximum(abs, $wSₙ), 1e-7]), maximum([maximum(abs, $wSₙ), 1e-7]))

# title_str = @lift "Time: $(round(T_data.times[$n] / 86400, digits=2)) days"
# Label(fig[0, :], title_str, tellwidth = false)

# hu = heatmap!(axu, yC, zC, uₙ, colormap=:RdBu_9, colorrange=ulim)
# hv = heatmap!(axv, yF, zC, vₙ, colormap=:RdBu_9, colorrange=vlim)
# hT = heatmap!(axT, yC, zC, Tₙ, colorrange=Tlim)
# hS = heatmap!(axS, yC, zC, Sₙ, colorrange=Slim)
# hν = heatmap!(axν, yC, zC, νₙ, colorrange=νlim)
# hκ = heatmap!(axκ, yC, zC, κₙ, colorrange=κlim)
# hRi = heatmap!(axRi, yC, zF, Riₙ, colormap=:RdBu_9, colorrange=(-20, 20))
# hwT = heatmap!(axwT, yC, zF, wTₙ, colormap=:RdBu_9, colorrange=wTlim)
# hwS = heatmap!(axwS, yC, zF, wSₙ, colormap=:RdBu_9, colorrange=wSlim)

# cbu = Colorbar(fig[1, 2], hu, label = "(m/s)")
# cbv = Colorbar(fig[1, 4], hv, label = "(m/s)")
# cbT = Colorbar(fig[2, 2], hT, label = "(°C)")
# cbS = Colorbar(fig[2, 4], hS, label = "(psu)")
# cbν = Colorbar(fig[1, 6], hν, label = "(m²/s)")
# cbκ = Colorbar(fig[2, 6], hκ, label = "(m²/s)")
# cbRi = Colorbar(fig[3, 6], hRi)
# cbwT = Colorbar(fig[3, 2], hwT, label = "(m/s °C)")
# cbwS = Colorbar(fig[3, 4], hwS, label = "(m/s psu)")

# tight_ticklabel_spacing!(cbu)
# tight_ticklabel_spacing!(cbv)
# tight_ticklabel_spacing!(cbT)
# tight_ticklabel_spacing!(cbS)
# tight_ticklabel_spacing!(cbν)
# tight_ticklabel_spacing!(cbκ)
# tight_ticklabel_spacing!(cbRi)
# tight_ticklabel_spacing!(cbwT)
# tight_ticklabel_spacing!(cbwS)

# CairoMakie.record(fig, "./$(dataname)_2D_sin_cooling_heating_23days_fluxes.mp4", 1:Nt, framerate=30) do nn
#     n[] = nn
# end
# #%%