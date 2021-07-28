# using Pkg
# pkg"add Oceananigans GLMakie"

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics
using Plots

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode

const Lx = 1000kilometers # east-west extent [m]
const Ly = 2000kilometers # north-south extent [m]
const Lz = 3kilometers    # depth [m]

Nx = 128
Ny = 2Nx
Nz = 32

s = 1.2 # stretching factor
z_faces(k) = - Lz * (1 - tanh(s * (k - 1) / Nz) / tanh(s))

arch = GPU()

grid = VerticallyStretchedRectilinearGrid(architecture = arch,
                                          topology = (Periodic, Bounded, Bounded),
                                          size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                          x = (-Lx/2, Lx/2),
                                          y = (0, Ly),
                                          z_faces = z_faces)

@info "Built a grid: $grid."

#=
plot(underlying_grid.Δzᵃᵃᶜ[1:Nz], underlying_grid.zᵃᵃᶜ[1:Nz],
     marker = :circle,
     ylabel = "Depth (m)",
     xlabel = "Vertical spacing (m)",
     legend = nothing)
=#

#####
##### Boundary conditions
#####

Qᵇ = 1e-8            # buoyancy flux magnitude [m² s⁻³]
y_shutoff = 5/6 * Ly # shutoff location for buoyancy flux [m]
τ = 1e-4             # surface kinematic wind stress [m² s⁻²]
μ = 1 / 30days       # bottom drag damping time-scale [s⁻¹]

#@inline buoyancy_flux(x, y, t, p) = ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0)
#buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, parameters=(Ly=grid.Ly, y_shutoff=y_shutoff, Qᵇ=Qᵇ))

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form=true, parameters=(Ly=grid.Ly, y_shutoff=y_shutoff, Qᵇ=Qᵇ))

# @inline u_stress(x, y, t, p) = - p.τ * sin(π * y / p.Ly)
# u_stress_bc = FluxBoundaryCondition(u_stress, parameters=(τ=τ, Ly=grid.Ly))

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return - p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=(τ=τ, Ly=grid.Ly))

#=
@inline u_drag(x, y, t, u, p) = - p.μ * p.Lz * u
@inline v_drag(x, y, t, v, p) = - p.μ * p.Lz * v

u_drag_bc = FluxBoundaryCondition(u_drag, field_dependencies=:u, parameters=(μ=μ, Lz = grid.Lz))
v_drag_bc = FluxBoundaryCondition(v_drag, field_dependencies=:v, parameters=(μ=μ, Lz = grid.Lz))
=#

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, 1] 
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=(μ=μ, Lz = grid.Lz))
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=(μ=μ, Lz = grid.Lz))

b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)
u_bcs = FieldBoundaryConditions(top = u_stress_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

#####
##### Coriolis
#####

coriolis = BetaPlane(latitude=-45)

#####
##### Forcing and initial condition
#####

const α = 1e-5 # [s⁻¹]
const f₀ = coriolis.f₀ # [s⁻¹]
const N² = 1e-5               # surface vertical buoyancy gradient [s⁻²]
const h = 1kilometer          # decay scale of stable stratification [m]

u_geostrophic(z) = α * (z + Lz)

@inline b_geostrophic(y) = - α * f₀ * y
@inline b_stratification(z) = N² * h * exp(z / h)

y_sponge = 19/20 * Ly # southern boundary of sponge layer [m]

@inline northern_mask(y, y_sponge, Ly) = max(0.0, y - y_sponge) / (Ly - y_sponge)
#@inline b_target(y, z) = b_geostrophic(y) + b_stratification(z)
@inline b_target(y, z) = b_stratification(z)

# We want this "continuous form" to work, but alas our attempts have failed so far.
#=
@inline b_relaxation(x, y, z, t, b, p) = - p.rate * northern_mask(y, p.y_sponge, p.Ly) * (b - b_target(y, z))

b_forcing = Forcing(b_relaxation,
                    field_dependencies = :b,
                    parameters = (rate=1/10days, y_sponge=y_sponge, Ly=grid.Ly))
=#

@inline function b_relaxation(i, j, k, grid, clock, model_fields, p)
    time = clock.time
    x = xnode(Center(), i, grid)
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    @inbounds b = model_fields.b[i, j, k]

    rate = 1 / 10days

    return - p.rate * northern_mask(y, p.y_sponge, p.Ly) * (b - b_target(y, z))
end

b_forcing = Forcing(b_relaxation,
                    discrete_form = true,
                    parameters = (rate=1/10days, y_sponge=y_sponge, Ly=grid.Ly))

#####
##### Dissipation
#####

#horizontal_diffusivity = AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

@show κ₂h = 1 #1e0 / day * grid.Δx^2 # [m² s⁻¹] horizontal viscosity and diffusivity
@show ν₂h = 30 #1e0 / day * grid.Δx^2 # [m² s⁻¹] horizontal viscosity and diffusivity
@show κ₄h = 1e-1 / day * grid.Δx^4 # [m⁴ s⁻¹] horizontal hyperviscosity and hyperdiffusivity

using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity

horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=ν₂h, κh=κ₂h)
#horizontal_diffusivity = HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0,
                                                                background_κz = 5e-6,
                                                                background_νz = 3e-4)
#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(architecture = arch,
                                    grid = grid,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = coriolis,
                                    closure = (horizontal_diffusivity, convective_adjustment),
                                    tracers = :b,
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (b=b_forcing,),
                                    )

@info "Built $model."

#####
##### Initial conditions
#####

g = model.free_surface.gravitational_acceleration # m s⁻²
u★ = 1e-8
b★ = 1e-9
δ = 0.2 * Ly
ϵu(x, y, z) = u★ * exp(-(y - Ly/2)^2 / 2δ^2) * randn()
ϵb(x, y, z) = b★ * randn()

# geostrophic initial condition
#ηᵢ(x, y) = - f₀ * α * Lz / 2g * (y - Ly/2) 
#uᵢ(x, y, z) = u_geostrophic(z) + ϵu(x, y, z)
#bᵢ(x, y, z) = b_geostrophic(y) + b_stratification(z)

# resting initial condition
ηᵢ(x, y) = 0
uᵢ(x, y, z) = 0
bᵢ(x, y, z) = b_stratification(z) + ϵb(x, y, z)

set!(model, u=uᵢ, b=bᵢ, η=ηᵢ)

#####
##### Simulation building
#####

wizard = TimeStepWizard(cfl=0.1, Δt=1minute, max_change=1.1, max_Δt=20minutes)

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
            prettytime(sim.Δt.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=60days, progress=print_progress, iteration_interval=10)

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
                                                        prefix = "eddying_channel",
                                                        force = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(5days),
                                                      prefix = "eddying_channel",
                                                      field_slicer = nothing,
                                                      verbose = true,
                                                      force = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
                                                        schedule = AveragedTimeInterval(1days, window=1days, stride=1),
                                                        prefix = "eddying_channel_averages",
                                                        verbose = true,
                                                        force = true)

@info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch
end

#####
##### Visualization
#####

grid = VerticallyStretchedRectilinearGrid(architecture = CPU(),
                                          topology = (Periodic, Bounded, Bounded),
                                          size = (grid.Nx, grid.Ny, grid.Nz),
                                          halo = (3, 3, 3),
                                          x = (-grid.Lx/2, grid.Lx/2),
                                          y = (0, grid.Ly),
                                          z_faces = z_faces)

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)

j′ = round(Int, grid.Ny / 2)
y′ = yζ[j′]

b_timeseries = FieldTimeSeries("eddying_channel.jld2", "b", grid=grid)
ζ_timeseries = FieldTimeSeries("eddying_channel.jld2", "ζ", grid=grid)
w_timeseries = FieldTimeSeries("eddying_channel.jld2", "w", grid=grid)

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
    
    @show bmax = maximum(abs, b_xy)
    @show ζmax = maximum(abs, ζ_xy)
    @show wmax = maximum(abs, w_xz)

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

mp4(anim, "eddying_channel.mp4", fps = 8) # hide
