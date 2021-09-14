# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using Plots
using Printf

const Lx = 400kilometers # east-west extent [m]
const Ly = 600kilometers # north-south extent [m]
const Lz = 1.8kilometers # depth [m]

Nx = Ny = 64
Nz = 16

σ = 1.2 # stretching factor
hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))


grid = VerticallyStretchedRectilinearGrid(size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                             x = (-Lx/2, Lx/2),
                                             y = (-Ly/2, Ly/2),
                                       z_faces = hyperbolically_spaced_faces,
                                      topology = (Bounded, Bounded, Bounded))

plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
      marker = :circle,
      ylabel = "Depth (m)",
      xlabel = "Vertical spacing (m)",
      legend = nothing)

α  = 2e-4 # [K⁻¹] thermal expansion coefficient 
g  = 9.81 # [m s⁻²] gravitational constant
cᵖ = 3991 # [J K⁻¹ kg⁻¹] heat capacity for seawater
ρ₀ = 1028 # [kg m⁻³] reference density

parameters = (Ly = Ly,
              Lz = Lz,
               τ = 0.1 / ρ₀,     # surface kinematic wind stress [m² s⁻²]
               μ = 1 / 180days,  # bottom drag damping time-scale [s⁻¹]
              Δb = 30 * α * g,   # surface vertical buoyancy gradient [s⁻²]
               λ = 30days        # relaxation time scale [s]
              )

# ## Boundary conditions
#
# ### Wind stress

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return - p.τ * cos(2π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

# ### Bottom drag

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, 1] 
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)


# ### Buoyancy relaxation

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
   y = ynode(Center(), j, grid)
   b = @inbounds model_fields.b[i, j, k]
   return - 1 / p.λ * (b - p.Δb * y / p.Ly)
end

Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

# ## Turbulence closure
closure = AnisotropicDiffusivity(νh=1000, νz=1e-2, κh=500, κz=1e-2)

# ## Model building

model = HydrostaticFreeSurfaceModel(architecture = CPU(),
                                    grid = grid,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = BetaPlane(latitude=45),
                                    closure = (closure,),
                                    tracers = :b,
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    forcing = (b=Fb,)
                                    )

# ## Initial conditions

bᵢ(x, y, z) = parameters.Δb * (1 + z / grid.Lz)

set!(model, b=bᵢ)

# ## Simulation setup

max_Δt = 1 / 5model.coriolis.f₀

wizard = TimeStepWizard(cfl=0.2, Δt=hour/10, max_change=1.1, max_Δt=max_Δt)

# Finally, we set up and run the the simulation.

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   maximum(abs, simulation.model.velocities.u),
                   maximum(abs, simulation.model.velocities.v),
                   maximum(abs, simulation.model.velocities.w),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=365days, iteration_interval=10, progress=print_progress)

# ## Output

u, v, w = model.velocities
b = model.tracers.b

speed = ComputedField(u^2 + v^2)
buoyancy_variance = ComputedField(b^2)

outputs = merge(model.velocities, model.tracers, (speed=speed, b²=buoyancy_variance))

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(2days),
                                                      prefix = "double_gyre",
                                                      field_slicer = FieldSlicer(k=model.grid.Nz),
                                                      force = true)

barotropic_u = AveragedField(model.velocities.u, dims=3)
barotropic_v = AveragedField(model.velocities.v, dims=3)

simulation.output_writers[:barotropic_velocities] =
    JLD2OutputWriter(model, (u=barotropic_u, v=barotropic_v),
                     schedule = AveragedTimeInterval(30days, window=10days),
                     prefix = "double_gyre_circulation",
                     force = true)

run!(simulation)

# # A neat movie

x, y, z = nodes(model.velocities.u)

xlims = (-grid.Lx/2 * 1e-3, grid.Lx/2 * 1e-3)
ylims = (-grid.Ly/2 * 1e-3, grid.Ly/2 * 1e-3)

x_km = x * 1e-3
y_km = y * 1e-3

nothing # hide

# Next, we open the JLD2 file, and extract the iterations we ended up saving at,

using JLD2, Plots

file = jldopen("double_gyre.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

# These utilities are handy for calculating nice contour intervals:

""" Returns colorbar levels equispaced from `(-clim, clim)` and encompassing the extrema of `c`. """
function divergent_levels(c, clim, nlevels=21)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    return ((-clim, clim), clim > cmax ? levels : levels = vcat([-cmax], levels, [cmax]))
end

""" Returns colorbar levels equispaced between `clims` and encompassing the extrema of `c`."""
function sequential_levels(c, clims, nlevels=20)
    levels = range(clims[1], stop=clims[2], length=nlevels)
    cmin, cmax = minimum(c), maximum(c)
    cmin < clims[1] && (levels = vcat([cmin], levels))
    cmax > clims[2] && (levels = vcat(levels, [cmax]))
    return clims, levels
end

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)
    
    @info "Drawing frame $i from iteration $iter \n"

    t = file["timeseries/t/$iter"]
    u = file["timeseries/u/$iter"][:, :, 1]
    v = file["timeseries/v/$iter"][:, :, 1]
    s = file["timeseries/speed/$iter"][:, :, 1]

    ulims, ulevels = divergent_levels(u, 1.0)
    slims, slevels = sequential_levels(s, (0.0, 1.5))

    kwargs = (aspectratio=:equal, linewidth=0, xlims=xlims,
              ylims=ylims, xlabel="x (km)", ylabel="y (km)")

    u_plot = contourf(x_km, y_km, u';
                      color = :balance,
                      clims = ulims,
                      levels = ulevels,
                      kwargs...)
                        
    s_plot = contourf(x_km, y_km, s';
                      color = :thermal,
                      clims = slims,
                      levels = slevels,
                      kwargs...)
                             
    plot(u_plot, s_plot, size=(800, 500),
         title = ["u(t="*string(round(t/day, digits=1))*" day)" "speed"])

    iter == iterations[end] && close(file)
end

gif(anim, "double_gyre.gif", fps = 8) # hide

# Plot the barotropic circulation

xv, yv, zv = nodes(model.velocities.v)

xv_km, yv_km = xv * 1e-3, yv * 1e-3

file = jldopen("double_gyre_circulation.jld2")

last_iteration = parse(Int, last(keys(file["timeseries/t"])))

U = file["timeseries/u/$last_iteration"][:, :, 1]
V = file["timeseries/v/$last_iteration"][:, :, 1]

U_plot = contourf(x_km, y_km, U', xlims=xlims, ylims=ylims,
                  linewidth=0, color=:balance, aspectratio=:equal)

V_plot = contourf(xv_km, yv_km, V', xlims=xlims, ylims=ylims,
                  linewidth=0, color=:balance, aspectratio=:equal)

plot(U_plot, V_plot, size=(800, 500),
     title=["Depth- and time-averaged \$ u \$" "Depth- and time-averaged \$ v \$"])

savefig("double_gyre_circulation.png") # hide
