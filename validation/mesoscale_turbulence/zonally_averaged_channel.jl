#using Pkg
# pkg"add Oceananigans CairoMakie JLD2"
ENV["GKSwstype"] = "100"
# pushfirst!(LOAD_PATH, @__DIR__)
# pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "..")) # add Oceananigans

using Printf
using Statistics
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using Oceananigans.Grids: xnode, ynode, znode

using Random
Random.seed!(1234)

filename = "zonally_averaged_channel_withGM"

# Architecture
architecture = CPU()

# Domain
const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = 2kilometers    # depth [m]

# number of grid points
Ny = 256
Nz = 80

save_fields_interval = 7days
stop_time = 5years
stop_time = 70 days
Δt₀ = 5minutes


# stretched grid

# we implement here a linearly streched grid in which the top grid cell has Δzₜₒₚ
# and every other cell is bigger by a factor σ, e.g.,
# Δzₜₒₚ, Δzₜₒₚ * σ, Δzₜₒₚ * σ², ..., Δzₜₒₚ * σᴺᶻ⁻¹,
# so that the sum of all cell heights is Lz

# Given Lz and stretching factor σ > 1 the top cell height is Δzₜₒₚ = Lz * (σ - 1) / σ^(Nz - 1)

# σ = 1.1 # linear stretching factor
# Δz_center_linear(k) = Lz * (σ - 1) * σ^(Nz - k) / (σ^Nz - 1) # k=1 is the bottom-most cell, k=Nz is the top cell
# linearly_spaced_faces(k) = k==1 ? -Lz : - Lz + sum(Δz_center_linear.(1:k-1))

# We choose a regular grid though because of numerical issues that yet need to be resolved
grid = RectilinearGrid(architecture,
                       topology = (Flat, Bounded, Bounded),
                       size = (Ny, Nz),
                       halo = (3, 3),
                       y = (0, Ly),
                       z = (-Lz, 0))

# The vertical spacing versus depth for the prescribed grid
#=
plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
     axis=(xlabel = "Vertical spacing (m)",
           ylabel = "Depth (m)"))
=#

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m s⁻²] gravitational constant
cᵖ = 3994.0   # [J K⁻¹] heat capacity
ρ  = 1024.0   # [kg m⁻³] reference density

parameters = (Ly = Ly,  
              Lz = Lz,    
              Qᵇ = 10 / (ρ * cᵖ) * α * g,          # buoyancy flux magnitude [m² s⁻³]    
              y_shutoff = 5/6 * Ly,                # shutoff location for buoyancy flux [m]
              τ = 0.2/ρ,                           # surface kinematic wind stress [m² s⁻²]
              μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
              ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
              H = Lz,                              # domain depth [m]
              h = 1000.0,                          # exponential decay scale of stable stratification [m]
              y_sponge = 19/20 * Ly,               # southern boundary of sponge layer [m]
              λt = 7.0days                         # relaxation time scale [s]
)

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form=true, parameters=parameters)


@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return - p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, 1] 
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)

b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

#####
##### Coriolis
#####

const f = -1e-4     # [s⁻¹]
const β =  1e-11    # [m⁻¹ s⁻¹]
coriolis = BetaPlane(f₀ = f, β = β)

#####
##### Forcing and initial condition
#####

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, j, k]
    return - 1 / timescale  * mask(y, p) * (b - target_b)
end

Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

# Turbulence closures

κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 30.0   # [m²/s] horizontal viscocity
κz = 0.5e-5 # [m²/s] vertical diffusivity
νz = 3e-4   # [m²/s] vertical viscocity


vertical_closure = ScalarDiffusivity(ν = νv,
                                     κ = κv,
                                     isotropy = ZDirection())

horizontal_closure = ScalarDiffusivity(ν = νh,
                                       κ = κh,
                                       isotropy = XYDirections())

diffusive_closure = (horizontal_closure, vertical_closure)
                                       
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

catke = CATKEVerticalDiffusivity()

gerdes_koberle_willebrand_tapering = Oceananigans.TurbulenceClosures.FluxTapering(1e-2)

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = 1000,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
#####
##### Model building
#####

@info "Building a model..."


model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = coriolis,
                                    #closure = (horizontal_diffusivity, convective_adjustment, gent_mcwilliams_diffusivity),
                                    closure = (catke, diffusive_closure..., gent_mcwilliams_diffusivity),
                                    tracers = (:b, :e, :c),
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (; b=Fb))

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * ( exp(z/parameters.h) - exp(-Lz/parameters.h) ) / (1 - exp(-Lz/parameters.h)) + ε(1e-8)
uᵢ(x, y, z) = ε(1e-8)
vᵢ(x, y, z) = ε(1e-8)
wᵢ(x, y, z) = ε(1e-8)

Δy = 100kilometers
Δz = 100
Δc = 2Δy
cᵢ(x, y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, b=bᵢ, u=uᵢ, v=vᵢ, w=wᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))


#####
##### Diagnostics
#####

u, v, w = model.velocities
b, c = model.tracers.b, model.tracers.c

dependencies = (gent_mcwilliams_diffusivity,
                b,
                Val(1),
                model.clock,
                model.diffusivity_fields,
                model.tracers,
                model.buoyancy,
                model.velocities)

using Oceananigans.TurbulenceClosures: diffusive_flux_y, diffusive_flux_z, ∇_dot_qᶜ

vb_op  = KernelFunctionOperation{Center, Face, Center}(diffusive_flux_y, grid, architecture=architecture, computed_dependencies=dependencies)
wb_op  = KernelFunctionOperation{Center, Center, Face}(diffusive_flux_z, grid, architecture=architecture, computed_dependencies=dependencies)
∇_q_op = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, grid, architecture=architecture, computed_dependencies=dependencies)

vb = Field(vb_op)
wb = Field(wb_op)
∇_q = Field(∇_q_op)

outputs = merge(fields(model), (; vb, wb, ∇_q))

# #####
# ##### Build checkpointer and output writer
# #####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(5years),
                                                        prefix = filename,
                                                        force = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(save_fields_interval),
                                                      prefix = filename,
                                                      field_slicer = nothing,
                                                      verbose = false,
                                                      force = true)

@info "Running the simulation..."

run!(simulation, pickup=false)

#####
##### Visualization
#####

using CairoMakie

filepath = filename * ".jld2"

zonal_file = jldopen(filepath)

grid = zonal_file["serialized/grid"]

xu, yu, zu = nodes((Face, Center, Center), grid)
xv, yv, zv = nodes((Center,Face,  Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)

u_timeseries = FieldTimeSeries(filepath, "u", grid=grid)
@show umax = maximum(abs, u_timeseries[:, :, :, :])

v_timeseries = FieldTimeSeries(filepath, "v", grid=grid)
@show umax = maximum(abs, v_timeseries[:, :, :, :])

w_timeseries = FieldTimeSeries(filepath, "w", grid=grid)
@show umax = maximum(abs, w_timeseries[:, :, :, :])

b_timeseries = FieldTimeSeries(filepath, "b", grid=grid)
@show b_timeseries

vb_timeseries = FieldTimeSeries(filepath, "vb", grid=grid)
@show vb_timeseries

wb_timeseries = FieldTimeSeries(filepath, "wb", grid=grid)
@show wb_timeseries

wb_timeseries = FieldTimeSeries(filepath, "wb", grid=grid)
@show wb_timeseries

∇_q_timeseries = FieldTimeSeries(filepath, "∇_q", grid=grid)
@show ∇_q_timeseries

_, _, _, nt = size(b_timeseries)

y_limits = (0, 2000)
z_limits = (-200, 0)

which_iteration = nt-1

fig, ax, hm = heatmap(yc * 1e-3, zc, interior(∇_q_timeseries)[1, :, :, which_iteration],
                      colormap=:balance, colorrange=(-1e-7, 1e-7),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "∇⋅q", textsize=30)
xlims!(y_limits[1], y_limits[2])
ylims!(z_limits[1], z_limits[2])
display(fig)

fig, ax, hm = heatmap(yv * 1e-3, zv, interior(vb_timeseries)[1, :, :, which_iteration],
                      colormap=:balance, colorrange=(-2e-5, 2e-5),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "v'b'", textsize=30)
xlims!(y_limits[1], y_limits[2])
ylims!(z_limits[1], z_limits[2])
display(fig)

fig, ax, hm = heatmap(yw * 1e-3, zw, interior(wb_timeseries)[1, :, :, which_iteration],
                      colormap=:balance, colorrange=(-1e-6, 1e-6),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "w'b'", textsize=30)
xlims!(y_limits[1], y_limits[2])
ylims!(z_limits[1], z_limits[2])
display(fig)

b = b_timeseries[which_iteration]

fig, ax, hm = heatmap(yc * 1e-3, zc, interior(b)[1, :, :],
                    #   colormap=:balance, colorrange=(-1e-7, 1e-7),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "b", textsize=30)
xlims!(y_limits[1], y_limits[2])
ylims!(z_limits[1], z_limits[2])
display(fig)


fig, ax, hm = heatmap(yu * 1e-3, zu, interior(u_timeseries)[1, :, :, which_iteration],
                      colormap=:balance, colorrange=(-0.5, 0.5),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "u", textsize=30)
# xlims!(y_limits[1], y_limits[2])
# ylims!(z_limits[1], z_limits[2])
display(fig)



fig, ax, hm = heatmap(yv * 1e-3, zv, interior(v_timeseries)[1, :, :, which_iteration],
                      colormap=:balance, colorrange=(-0.1, 0.1),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "v", textsize=30)
# xlims!(y_limits[1], y_limits[2])
# ylims!(z_limits[1], z_limits[2])
display(fig)



fig, ax, hm = heatmap(yw * 1e-3, zw, interior(w_timeseries)[1, :, :, which_iteration],
                      colormap=:balance, colorrange=(-1e-4, 1e-4),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "w", textsize=30)
# xlims!(y_limits[1], y_limits[2])
# ylims!(z_limits[1], z_limits[2])
display(fig)




b_z = Field(Center, Center, Face, grid)

b_z .= ∂z(b)

fig, ax, hm = heatmap(yw * 1e-3, zw, interior(b_z)[1, :, :],
                      colormap=:curl, colorrange=(-5e-4, 5e-4),
                      axis=(xlabel="y", ylabel="z", xticklabelsize=20,  yticklabelsize=20, xlabelsize=25, ylabelsize=25))
Colorbar(fig[1, 2], hm, ticklabelsize=20)
fig[0, :] = Label(fig, "∂b/∂z", textsize=30)
xlims!(y_limits[1], y_limits[2])
ylims!(z_limits[1], z_limits[2])
display(fig)



umax = 1
ulims = (-umax, umax) .* 0.8
ulevels = vcat([-umax], range(ulims[1], ulims[2], length=31), [umax])

ylims = (0, grid.Ly) .* 1e-3
zlims = (-grid.Lz, 0)

anim = @animate for i in 1:length(b_timeseries.times)-1
    b = b_timeseries[i]
    u = u_timeseries[i]
    
    b_z .= ∂z(b)

    b_yz = interior(b)[1, :, :]
    u_yz = interior(u)[1, :, :]
   
    b_z_yz = interior(b_z)[1, :, :]
    
    @show bmax = max(1e-9, maximum(abs, b_yz))

    blims = (-bmax, bmax) .* 0.8
    
    blevels = vcat([-bmax], range(blims[1], blims[2], length=31), [bmax])
    
    u_yz_plot = contourf(yu * 1e-3, zu, u_yz',
                         xlabel = "y (km)",
                         ylabel = "z (m)",
                         aspectratio = :equal,
                         linewidth = 0,
                         levels = ulevels,
                         clims = ulims,
                         xlims = ylims,
                         ylims = zlims,
                         color = :balance)
    
    bz_yz_plot = heatmap(yw * 1e-3, zw, b_z_yz',
                         xlabel = "y (km)",
                         ylabel = "z (m)",
                         aspectratio = :equal,
                         xlims = ylims,
                         ylims = zlims,
                         color = :balance)
    
    contour!(u_yz_plot,
             yc * 1e-3, zc, b_yz',
             linewidth = 1,
             color = :black,
             levels = blevels,
             legend = :none)
end

mp4(anim, filename*".mp4", fps = 8) # hide
