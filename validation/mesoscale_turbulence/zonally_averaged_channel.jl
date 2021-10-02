# using Pkg
# pkg"add Oceananigans GLMakie"

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Printf
using Statistics
using Plots

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

# number of grid points
Nx = 1
Ny = 400
Nz = 35

# stretched grid 
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)
const Lz = sum(Δz_center)
z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0

arch = CPU()
FT = Float64

grid = VerticallyStretchedRectilinearGrid(architecture = arch,
                                          topology = (Periodic, Bounded, Bounded),
                                          size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                          x = (0, Lx),
                                          y = (0, Ly),
                                          z_faces = z_faces)

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m/s²] gravitational constant
cᵖ = 3994.0   # [J/K]  heat capacity
ρ  = 1024.0   # [kg/m³] reference density

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

# ynode(::Type{Center}, j, grid::RegularRectilinearGrid) = @inbounds grid.yC[j]
# ynode(::Type{Center}, j, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.yᵃᵃᶜ[j]


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
coriolis = BetaPlane(FT, f₀ = f, β = β)

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

horizontal_diffusivity = AnisotropicDiffusivity(νh=νh, νz=νz, κh=κh, κz=κz)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

gerdes_koberle_willebrand_tapering = Oceananigans.TurbulenceClosures.FluxTapering(1e-2)

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = 100, κ_symmetric = 100,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
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
                                    closure = (horizontal_diffusivity, convective_adjustment, gent_mcwilliams_diffusivity),
                                    tracers = :b,
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (; b=Fb))

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

wizard = TimeStepWizard(cfl=0.1, Δt=5minutes, max_change=1.1, max_Δt=20minutes)

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
 #           prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=2days, progress=print_progress, iteration_interval=10)

#####
##### Diagnostics
#####

u, v, w = model.velocities
b = model.tracers.b

outputs = (; b, u)

# #####
# ##### Build checkpointer and output writer
# #####

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(0.2days),
                                                      prefix = "zonally_averaged_channel",
                                                      field_slicer = nothing,
                                                      verbose = true,
                                                      force = true)

@info "Running the simulation..."

run!(simulation, pickup=false)

#####
##### Visualization
#####

grid = VerticallyStretchedRectilinearGrid(architecture = CPU(),
                                          topology = (Periodic, Bounded, Bounded),
                                          size = (grid.Nx, grid.Ny, grid.Nz),
                                          halo = (3, 3, 3),
                                          x = (0, grid.Lx),
                                          y = (0, grid.Ly),
                                          z_faces = z_faces)

xu, yu, zu = nodes((Face, Center, Center), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)

u_timeseries = FieldTimeSeries("zonally_averaged_channel.jld2", "u", grid=grid)
@show umax = maximum(abs, u_timeseries[:, :, :, :])
    
b_timeseries = FieldTimeSeries("zonally_averaged_channel.jld2", "b", grid=grid)
@show b_timeseries

ulims = (-umax, umax) .* 0.8
ulevels = vcat([-umax], range(ulims[1], ulims[2], length=31), [umax])

ylims = (0, grid.Ly) .* 1e-3
zlims = (-grid.Lz, 0)

anim = @animate for i in 1:length(b_timeseries.times)
    b = b_timeseries[i]
    u = u_timeseries[i]
    
    b_yz = interior(b)[1, :, :]
    u_yz = interior(u)[1, :, :]
    
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
    
    contour!(u_yz_plot,
             yc * 1e-3, zc, b_yz',
             linewidth = 1,
             color = :black,
             levels = blevels,
             legend = :none)
end

mp4(anim, "zonally_averaged_channel.mp4", fps = 8) # hide

