using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.Operators
using Oceananigans.Operators: ℑxyz
using Statistics
using Random
using JLD2
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: getdiffusivity
using Oceananigans.Models.HydrostaticFreeSurfaceModels: Fields
using Oceananigans
using Oceananigans.Units
using GLMakie

######################################################################################
# Barolinic adjustment
filename = "test_baroclinic_adjustment_" * gradient

# Architecturz
architecture = CPU()

# Domain
Lx = 1000kilometers # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 4kilometers     # depth [m]

Nx, Ny = 96, 96
Nz = 40

# first try

save_fields_interval = 10day
save_volumes_interval = 100day
save_tracers_interval = 10day
stop_time = 1000day
Δt = 10minutes
Nts = 20

# tchebytchev ?
#zfaces = -Lz .+ (-0.5 * (cos.(range(0.0, π, length=Nz + 1))) .+ 0.5) .* Lz # range(-Lz, 0.0, length = Nz+1)
#zfaces = range(-Lz, 0.0, length=Nz + 1)


# 3D channel periodic in x direction
#default halo
grid = RectilinearGrid(architecture;
                       size = (Ny, Ny, Nz), 
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz,0),
                       topology = (Periodic, Bounded, Bounded),
                       halo=(3,3,3)) 

coriolis = BetaPlane(latitude = -45)

######################################################################################
# visualize the z-grid 

#set_theme!(Theme(fontsize=24, linewidth=3))
#fig = Figure(size=(200,300))
#ax = (xlabel = "zgrid", ylabel="z-spacing")
#z= znodes(grid,Center())
#Δz = zspacings(grid,Center())

# Diffusivities and closures coefficients
gm_parameters = (max_C = 20, 
                 K₀ᴳᴹ =  1e3,
                 β = coriolis.β)

# Custom GM coefficient, function of `i, j, k, grid, clock, fields, parameters`
@inline function κskew(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, p) 
    b = fields.b

    ∂ʸb = ℑxyz(i, j, k, grid, (Center(), Face(), Center()), (ℓx, ℓy, ℓz), ∂yᶜᶠᶜ, b)
    ∂ᶻb = ℑxyz(i, j, k, grid, (Center(), Center(), Face()), (ℓx, ℓy, ℓz), ∂zᶜᶜᶠ, b)

    Sʸ = ∂ʸb / ∂ᶻb 
    Sʸ = ifelse(isnan(Sʸ), zero(grid), Sʸ)

    z  = znode(k, grid, Center())
    C  = min(max(zero(grid), 1 - p.β / Sʸ * z), p.max_C)

    return p.K₀ᴳᴹ * C
end

κz = 1e-5
νz = 1e-5

vertical_closure = VerticalScalarDiffusivity(ν=νz, κ=κz)

gerdes_koberle_willebrand_tapering = FluxTapering(1e-1)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κskew,
                                                                κ_symmetric = namelist.κsymmetric,
                                                                skew_discrete_form = true,
                                                                skew_loc = (nothing, nothing, nothing),
                                                                parameters = gm_parameters,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering,
                                                                required_halo_size = 3)

closures = (vertical_closure, gent_mcwilliams_diffusivity)
 
model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = closures,
                                    tracers = (:b, :c),
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO())
                                
"""
    ramp(y, Δy)

Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:
```
            y < -Δy/2 => ramp = 0
    -Δy/2 < y < -Δy/2 => ramp = y / Δy
            y >  Δy/2 => ramp = 1
```
"""
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 500kilometers
Δz = 100

Δc = 2Δy # extend
Δb = Δy * M² #buoyancy variation with ramp
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()
# peridically in x : no gaussian distribution in x for initial condition 
cᵢ(λ, y, z) = exp(-y^2 / 2Δc^2) *  exp(-(z + Lz/4)^2 / 2Δz^2) # ou constante

set!(model, b=bᵢ, c=cᵢ)


### Visualize initial conditions

x, y, z = 1e-3 .* nodes(grid, (Center(), Center(), Center()))

#Figure()
#ax = (xlabel = "y [km]", ylabel="z [km]")
#b = model.tracers.b
#c = model.tracers.c

#fig, ax, hm = heatmap(y, z, Matrix(interior(c)[1, :, :]),
#                      colormap=:deep,
#                      axis = (xlabel = "y [km]",
#                              ylabel = "z [km]",
#                              title = "b(x=0, y, z, t=0)",
#                              titlesize = 24))
#Colorbar(fig[1, 2], hm, label = "[m s⁻²]")
#fig


#####
##### Simulation building
#####


simulation = Simulation(model; Δt, stop_time=stop_time)

# add timestep wizard callback
# wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=Δt)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
wall_clock = [time_ns()]

u, v, w = model.velocities

Ec   = (u^2 + v^2 + w^2)/2 
Eint = Field(Average(Ec, dims=(1, 2, 3)))

filename = "gmoutput"

@inline getmydiffusivity(i, j, k, grid, κ, location, clock, fields) = getdiffusivity(κ, i, j, k, grid, location, clock, fields)

κgm = KernelFunctionOperation{Center, Center, Center}(getmydiffusivity, grid, model.closure[2].κ_skew, (Center(), Center(), Center()), model.clock, fields(model))
κgm = Field(κgm)

function print_progress(sim)
    compute!(Eint)
    compute!(κgm)

    @info @sprintf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, Eint: %6.3e, κgm: (%6.3e %6.3e), next Δt: %s\n",
                    100 * (sim.model.clock.time / sim.stop_time),
                    sim.model.clock.iteration,
                    prettytime(sim.model.clock.time),
                    prettytime(1e-9 * (time_ns() - wall_clock[1])),
                    maximum(abs, u),
                    maximum(abs, v),
                    maximum(abs, w),
	                interior(Eint)[1, 1, 1],
                    maximum(κgm),
                    minimum(κgm),
                    prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(10))

simulation.output_writers[:scalars] = JLD2OutputWriter(model, model.tracers,
                                                       schedule = TimeInterval(save_tracers_interval),
                                                       filename = filename * "_tracers",
                                                       overwrite_existing = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(save_volumes_interval),
                                                      filename = filename * "_fields",
                                                      overwrite_existing = true)

# B = Field(Average(model.tracers.b, dims=1))
# C = Field(Average(model.tracers.c, dims=1))
# U = Field(Average(model.velocities.u, dims=1))
# V = Field(Average(model.velocities.v, dims=1))
# W = Field(Average(model.velocities.w, dims=1))

# simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B, c=C, u=U, v=V, w=W),
#                                                      schedule = TimeInterval(save_fields_interval),
#                                                      filename = filename * "_zonal_average",
#                                                      overwrite_existing = true)
@info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end

@info "Simulation completed in " * prettytime(simulation.run_wall_time)
