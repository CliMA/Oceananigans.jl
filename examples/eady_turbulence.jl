# # Eady turbulence example
#
# In this example, we initialize a random velocity field and observe its viscous,
# turbulent decay in a two-dimensional domain. This example demonstrates:
#
#   * How to use a tuple of turbulence closures
#   * How to use biharmonic diffusivity
#   * How to implement a background flow (a background geostrophic shear)

# # The grid

using Oceananigans

grid = RegularCartesianGrid(size=(64, 64, 16), x=(-1e5, 1e5), y=(-1e5, 1e5), z=(-1e3, 0))

# # Rotation
#
# The classical Eady problem is posed on an $f$-plane. We use a Coriolis parameter
# typical to mid-latitudes on Earth,

coriolis = FPlane(f=1e-4) # [s⁻¹]
                            
# # The background flow
#
# The Eady problem is non-linearized around a geostrophic flow
#
# $ ψ(y, z) = - α y (z + L_z) $,
#
# where $α$ is the geostrophic shear and horizontal buoyancy gradient
# and $L_z$ is the depth of the domain. The background buoyancy,
# including both the geostrophic flow component
# and a background stable stratification component, is
#
# $ b + N^2 z = f ∂_z ψ + N^2 z = - α f y + N^2 z$
#
# The background velocity field is
#
# $ u = - ∂_y ψ = α (z + L_z) $
#
# To impose this background flow with user-defined forcing functions we
# require parameters that specify the veritcal shear, the Coriolis parameter
# and the domain depth.

using Oceananigans.Fields: BackgroundField

background_parameters = (α=2.5e-4, f=coriolis.f, N=2.5e-3, Lz=grid.Lz)

# We can then build out background fields

U(x, y, z, t, p) = + p.α * (z + p.Lz)
B(x, y, z, t, p) = - p.α * p.f * y + p.N^2 * z

U_field = BackgroundField(U, parameters=background_parameters)
B_field = BackgroundField(B, parameters=background_parameters)

stratification = BackgroundField((x, y, z, t, p) -> p.N^2 * z, parameters=background_parameters)

# # Friction coefficients

Δh = grid.Lx / grid.Nx     # [meters] horizontal grid spacing for diffusivity calculations
Δz = grid.Lz / grid.Nz     # [meters] vertical grid spacing for diffusivity calculations

κ₂z = 1e-3 # Laplacian vertical diffusivity, [m² s⁻¹]
κ₄h = 1e-4 * (Δh / Δz)^4 * κ₂z # Biharmonic horizontal diffusivity, [m⁴ s⁻¹]

# # Boundary conditions
#
# These boundary conditions prescribe a linear drag at the bottom as a flux
# condition. We also fix the surface and bottom buoyancy to enforce a buoyancy
# gradient `N²`.

bc_parameters = (μ=1e-6, Lz=grid.Lz)

@inline τ₁₃(i, j, grid, clock, model_fields, params) = @inbounds params.μ * params.Lz * model_fields.u[i, j, 1]
@inline τ₂₃(i, j, grid, clock, model_fields, params) = @inbounds params.μ * params.Lz * model_fields.v[i, j, 1]

linear_drag_u = BoundaryCondition(Flux, τ₁₃, discrete_form=true, parameters=bc_parameters)
linear_drag_v = BoundaryCondition(Flux, τ₂₃, discrete_form=true, parameters=bc_parameters)

u_bcs = UVelocityBoundaryConditions(grid, bottom = linear_drag_u) 
v_bcs = VVelocityBoundaryConditions(grid, bottom = linear_drag_v)

using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂xᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑxᶜᵃᵃ, ℑxzᶠᵃᶜ

# The $x$-momentum forcing
#
# $ F_u = - α w - α (z + H) ∂ₓu $
#
# is applied at location `(f, c, c)`.

function Fu_eady_func(i, j, k, grid, clock, model_fields, p)
    return @inbounds (- p.α * ℑxzᶠᵃᶜ(i, j, k, grid, model_fields.w)
                      - p.α * (grid.zC[k] + p.Lz) * ∂xᶠᵃᵃ(i, j, k, grid, ℑxᶜᵃᵃ, model_fields.u))
end

# The $y$-momentum forcing
#
# $ F_v = - α (z + H) ∂ₓv
#
# is applied at location `(c, f, c)`.

function Fv_eady_func(i, j, k, grid, clock, model_fields, p)
    return @inbounds -p.α * (grid.zC[k] + p.Lz) * ∂xᶜᵃᵃ(i, j, k, grid, ℑxᶠᵃᵃ, model_fields.v)
end

# The $z$-momentum forcing
#
# $ F_w = - α (z + H) ∂ₓw
#
# is applied at location `(c, c, f)`.

function Fw_eady_func(i, j, k, grid, clock, model_fields, p)
    return @inbounds -p.α * (grid.zF[k] + p.Lz) * ∂xᶜᵃᵃ(i, j, k, grid, ℑxᶠᵃᵃ, model_fields.w)
end

# The buoyancy forcing
#
# $ F_b = - α (z + H) ∂ₓb + α f v $
#
# is applied at location `(c, c, c)`.

function Fb_eady_func(i, j, k, grid, clock, model_fields, p)
    return @inbounds (- p.α * (grid.zC[k] + p.Lz) * ∂xᶜᵃᵃ(i, j, k, grid, ℑxᶠᵃᵃ, model_fields.b)
                      + p.f * p.α * ℑyᵃᶜᵃ(i, j, k, grid, model_fields.v))
end

Fu_eady = Forcing(Fu_eady_func, parameters=background_parameters, discrete_form=true)
Fv_eady = Forcing(Fv_eady_func, parameters=background_parameters, discrete_form=true)
Fw_eady = Forcing(Fw_eady_func, parameters=background_parameters, discrete_form=true)
Fb_eady = Forcing(Fb_eady_func, parameters=background_parameters, discrete_form=true)

# # Turbulence closures
#
# We use a horizontal biharmonic diffusivity and a Laplacian vertical diffusivity
# to dissipate energy in the Eady problem.
# To use both of these closures at the same time, we set the keyword argument
# `closure` a tuple of two closures. Note that the "2D Leith" parameterization may
# also be a sensible choice to pair with a Laplacian vertical diffusivity for this problem.

closure = (AnisotropicDiffusivity(νh=0, κh=0, νz=κ₂z, κz=κ₂z),
           AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h))
           #TwoDimensionalLeith())

# # Model instantiation

using Oceananigans.Advection: WENO5

model = IncompressibleModel(               grid = grid,
                                   architecture = CPU(),
                                      advection = WENO5(),
                                    timestepper = :RungeKutta3,
                                       coriolis = coriolis,
                                       buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                              background_fields = (b=B_field, u=U_field),
                              #background_fields = (b=stratification,),
                              #          forcing = (u=Fu_eady, v=Fv_eady, w=Fw_eady, b=Fb_eady),
                                        closure = closure,
                            boundary_conditions = (u=u_bcs, v=v_bcs))

# # Initial conditions
#
# For initial conditions we impose a linear stratifificaiton with some
# random noise.

## A noise function, damped at the boundaries
Ξ(z) = rand() * z/grid.Lz * (z/grid.Lz + 1)

## Buoyancy: linear stratification plus noise
u₀(x, y, z) = 1e-1 * background_parameters.α * grid.Lz * Ξ(z)

set!(model, u=u₀)

# # The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the CFL close to a
# desired value.

using Oceananigans.Utils: minute, hour, day

wizard = TimeStepWizard(cfl=0.5, Δt=10minute, max_change=1.1, max_Δt=10minute)

# Set up the simulation with a progress messenger

using Printf
using Oceananigans.Diagnostics: AdvectiveCFL

CFL = AdvectiveCFL(wizard)

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(1e-9 * (time_ns() - start_time)),
                        prettytime(sim.Δt.Δt),
                        CFL(sim.model))

simulation = Simulation(model, Δt = wizard, iteration_interval = 10,
                                                stop_iteration = 1000,
                                                      progress = progress)

using Oceananigans.Diagnostics: NaNChecker

push!(simulation.diagnostics, NaNChecker(model; iteration_interval=1, fields=Dict(:u=>model.velocities.u)))

run!(simulation)

# ## Custom output
#
# We also create objects for computing the vertical vorticity and divergence
# for plotting purposes.

using Oceananigans.OutputWriters, Oceananigans.AbstractOperations
using Oceananigans.Fields: ComputedField

u, v, w = model.velocities

ζ = ComputedField(∂x(v) - ∂y(u))
δ = ComputedField(-∂z(w))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model,
                     merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                  prefix = "eady_turbulence",
                      iteration_interval = 5,
                                   force = true)

simulation.stop_iteration += 400
run!(simulation)

# Finally, we animate the results by opening the JLD2 file, extract the
# iterations we ended up saving at, and plot the evolution of the
# temperature profile in a loop over the iterations.

using JLD2, Plots, Printf, Oceananigans.Grids
using Oceananigans.Grids: x_domain, y_domain, z_domain

pyplot()

xζ, yζ, zζ = nodes(ζ)
xδ, yδ, zδ = nodes(δ)
xw, yw, zw = nodes(w)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

function nice_divergent_levels(c, clim)
    levels = range(-clim, stop=clim, length=10)

    cmax = maximum(abs, c)

    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=10), [cmax])
    end

    return levels
end

@info "Making an animation from saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from file
    ζ = file["timeseries/ζ/$iter"][:, :, grid.Nz]
    δ = file["timeseries/δ/$iter"][:, :, grid.Nz]
    w = file["timeseries/w/$iter"][:, 1, :]

    ζlim = max(0.5 * maximum(abs, ζ), 1e-9)
    δlim = max(0.5 * maximum(abs, δ), 1e-9)
    wlim = max(0.5 * maximum(abs, w), 1e-9)

    ζlevels = nice_divergent_levels(ζ, ζlim)
    δlevels = nice_divergent_levels(δ, δlim)
    wlevels = nice_divergent_levels(w, wlim)

    ζ_plot = contourf(xζ, yζ, ζ'; color = :balance,
                            aspectratio = :equal,
                                 legend = false,
                                  clims = (-ζlim, ζlim),
                                 levels = ζlevels,
                                  xlims = x_domain(grid),
                                  ylims = y_domain(grid),
                                 xlabel = "x (m)",
                                 ylabel = "y (m)")
    
    δ_plot = contourf(xδ, yδ, δ'; color = :balance,
                            aspectratio = :equal,
                                 legend = false,
                                  clims = (-δlim, δlim),
                                 levels = δlevels,
                                  xlims = x_domain(grid),
                                  ylims = y_domain(grid),
                                 xlabel = "x (m)",
                                 ylabel = "y (m)")

    w_plot = contourf(xw / 1e2, zw, w'; color = :balance,
                            aspectratio = :equal,
                                 legend = false,
                                  clims = (-wlim, wlim),
                                 levels = wlevels,
                                  xlims = x_domain(grid) ./ 1e2,
                                  ylims = z_domain(grid),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

    plot(ζ_plot, δ_plot, w_plot, layout=(1, 3), size=(2000, 800),
         title = ["ζ(x, y, z=0, t) (1/s)" "δ(x, y, z=0, z, t) (1/s)" "w(x, y=0, z, t) (m/s)"])

    iter == iterations[end] && close(file)
end

mp4(anim, "eady_turbulence.mp4", fps = 4) # hide

`open -a VLC eady_turbulence.mp4`
