#using Pkg
# pkg"add Oceananigans GLMakie"

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

# Architecture
architecture  = CPU()

# number of grid points
Nx = 100
Ny = 200
Nz = 35

# stretched grid 
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)

const Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0

grid = RectilinearGrid(architecture = architecture,
                       topology = (Periodic, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces)

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m/s²] gravitational constant
cᵖ = 3994.0   # [J/K]  heat capacity
ρ  = 999.8    # [kg/m³] reference density

parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10/(ρ * cᵖ) * α * g,            # buoyancy flux magnitude [m² s⁻³]    
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

const f = -1e-4
const β =  1e-11
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

# closure

κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 30.0   # [m²/s] horizontal viscocity
κz = 0.5e-5 # [m²/s] vertical diffusivity
νz = 3e-4   # [m²/s] vertical viscocity

closure = AnisotropicDiffusivity(νh=νh, νz=νz, κh=κh, κz=κz)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                    grid = grid,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENO5(grid = grid),
                                    tracer_advection = WENO5(grid = grid),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = coriolis,
                                    closure = (closure, convective_adjustment),
                                    tracers = :b,
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (; b=Fb,)
                                    )

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * ( exp(z/parameters.h) - exp(-Lz/parameters.h) ) / (1 - exp(-Lz/parameters.h)) + ε(1e-8)

set!(model, b=bᵢ)

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
b = model.tracers.b

ζ = ComputedField(∂x(v) - ∂y(u))

B = AveragedField(b, dims=1)
V = AveragedField(v, dims=1)
W = AveragedField(w, dims=1)

b′ = b - B
v′ = v - V
w′ = w - W

v′b′ = AveragedField(v′ * b′, dims=1)
w′b′ = AveragedField(w′ * b′, dims=1)

outputs = (; b, ζ, w)

averaged_outputs = (; v′b′, w′b′, B)

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(100days),
                                                        prefix = "abernathey_channel",
                                                        force = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(5days),
                                                      prefix = "abernathey_channel",
                                                      field_slicer = nothing,
                                                      verbose = true,
                                                      force = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
                                                        schedule = AveragedTimeInterval(1days, window=1days, stride=1),
                                                        prefix = "abernathey_channel_averages",
                                                        verbose = true,
                                                        force = true)

@info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end

# #####
# ##### Visualization
# #####

using Plots

grid = RectilinearGrid(architecture = CPU(),
                       topology = (Periodic, Bounded, Bounded),
                       size = (grid.Nx, grid.Ny, grid.Nz),
                       halo = (3, 3, 3),
                       x = (0, grid.Lx),
                       y = (0, grid.Ly),
                       z = z_faces)

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)

j′ = round(Int, grid.Ny / 2)
y′ = yζ[j′]

b_timeseries = FieldTimeSeries("abernathey_channel.jld2", "b", grid=grid)
ζ_timeseries = FieldTimeSeries("abernathey_channel.jld2", "ζ", grid=grid)
w_timeseries = FieldTimeSeries("abernathey_channel.jld2", "w", grid=grid)

@show b_timeseries

anim = @animate for i in 1:length(b_timeseries.times)
    b = b_timeseries[i]
    ζ = ζ_timeseries[i]
    w = w_timeseries[i]

    b′ = interior(b) .- mean(b)
    b_xy = b′[:, :, grid.Nz]
    ζ_xy = interior(ζ)[:, :, grid.Nz]
    ζ_xz = interior(ζ)[:, j′, :]
    w_xz = interior(w)[:, j′, :]

    @show bmax = max(1e-9, maximum(abs, b_xy))
    @show ζmax = max(1e-9, maximum(abs, ζ_xy))
    @show wmax = max(1e-9, maximum(abs, w_xz))

    blims = (-bmax, bmax) .* 0.8
    ζlims = (-ζmax, ζmax) .* 0.8
    wlims = (-wmax, wmax) .* 0.8

    blevels = vcat([-bmax], range(blims[1], blims[2], length=31), [bmax])
    ζlevels = vcat([-ζmax], range(ζlims[1], ζlims[2], length=31), [ζmax])
    wlevels = vcat([-wmax], range(wlims[1], wlims[2], length=31), [wmax])

    xlims = (-grid.Lx/2, grid.Lx/2) .* 1e-3
    ylims = (0, grid.Ly) .* 1e-3
    zlims = (-grid.Lz, 0)

    w_xz_plot = contourf(xw * 1e-3, zw, w_xz',
                        xlabel = "x (km)",
                        ylabel = "z (m)",
                        aspectratio = 0.05,
                        linewidth = 0,
                        levels = wlevels,
                        clims = wlims,
                        xlims = xlims,
                        ylims = zlims,
                        color = :balance)

    ζ_xy_plot = contourf(xζ * 1e-3, yζ * 1e-3, ζ_xy',
                        xlabel = "x (km)",
                        ylabel = "y (km)",
                        aspectratio = :equal,
                        linewidth = 0,
                        levels = ζlevels,
                        clims = ζlims,
                        xlims = xlims,
                        ylims = ylims,
                        color = :balance)
    
    b_xy_plot = contourf(xc * 1e-3, yc * 1e-3, b_xy',
                        xlabel = "x (km)",
                        ylabel = "y (km)",
                        aspectratio = :equal,
                        linewidth = 0,
                        levels = blevels,
                        clims = blims,
                        xlims = xlims,
                        ylims = ylims,
                        color = :balance)

    w_xz_title = @sprintf("w(x, z) at t = %s", prettytime(ζ_timeseries.times[i]))
    ζ_xz_title = @sprintf("ζ(x, z) at t = %s", prettytime(ζ_timeseries.times[i]))
    ζ_xy_title = "ζ(x, y)"
    b_xy_title = "b(x, y)"

    layout = @layout [upper_slice_plot{0.2h}
                    Plots.grid(1, 2)]
    
    plot(w_xz_plot, ζ_xy_plot,  b_xy_plot, layout = layout, size = (1200, 1200), title = [w_xz_title ζ_xy_title b_xy_title])
end

mp4(anim, "abernathey_channel.mp4", fps = 8) # hide
