# # Eady turbulence example
#
# In this example, we initialize a random velocity field and observe its viscous,
# turbulent decay in a two-dimensional domain. This example demonstrates:
#
#   * How to use a tuple of turbulence closures
#   * How to use biharmonic diffusivity
#   * How to implement a background flow (a background geostrophic shear)

using Oceananigans, Oceananigans.Diagnostics, Oceananigans.OutputWriters,
      Oceananigans.AbstractOperations, Random, Printf

# These imports from Oceananigans's `TurbuleneClosures` module are needed for imposing
# a background flow as a user-defined forcing.

using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂xᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑxᶜᵃᵃ, ℑxzᶠᵃᶜ

using Oceananigans: Face, Cell

# # Parameters
#
# Here we define numerical and physical parameters for an Eady problem.
# We use a very fast baroclinic growth rate for the sake of this coarse 3D
# example.

Nh = 64           # horizontal resolution
Nz = 32           # vertical resolution
Lh = 2e6          # [meters] horizontal domain extent
Lz = 1e3          # [meters] vertical domain extent
Rᵈ = Lh / 10      # [m] Deformation radius
σᵇ = 0.1day       # [s] Growth rate for baroclinic instability
τᵏ = 1.0day       # [s] biharmonic / viscous damping timescale
 μ = 1/30day      # [s⁻¹] linear drag decay scale
 f = 1e-4         # [s⁻¹] Coriolis parameter

 Δh = Lh / Nh     # [meters] horizontal grid spacing for diffusivity calculations
 Δz = Lz / Nz     # [meters] vertical grid spacing for diffusivity calculations
κ₄h = Δh^4 / τᵏ   # [m⁴ s⁻¹] Biharmonic horizontal diffusivity
 κᵥ = Δz^2 / 20τᵏ # [m² s⁻¹] Laplacian vertical diffusivity

@show N² = (Rᵈ * f / Lz)^2      # [s⁻²] Initial buoyancy gradient
@show  α = sqrt(N²) / (f * σᵇ)  # [s⁻¹] background shear

end_time = 3day # Simulation end time

# # Boundary conditions
#
# These boundary conditions prescribe a linear drag at the bottom as a flux
# condition. We also fix the surface and bottom buoyancy to enforce a buoyancy
# gradient `N²`.

bc_parameters = (μ=μ, H=Lz)

@inline τ₁₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * p.H * U.u[i, j, 1]
@inline τ₂₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * p.H * U.v[i, j, 1]

ubcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₁₃_linear_drag))
vbcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₂₃_linear_drag))
bbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 0),
                               bottom = BoundaryCondition(Value, -N² * Lz))

# # Forcing functions
#
# The Eady problem is non-linearized around a geostrophic flow
#
# $ ψ = - α y (z + H) $,
#
# where α is the geostrophic shear and horizontal buoyancy gradient.
# The background buoyancy associated with this geostrophic flow is
#
# $ b = f ∂_z ψ = - α f y $
#
# To impose this background flow with user-defined forcing functions we
# require parameters that specify the veritcal shear, the Coriolis parameter
# and the domain depth.

forcing_parameters = (α=α, f=f, H=Lz)

# The $x$-momentum forcing
#
# $ F_u = - α w - α (z + H) ∂ₓu $
#
# is applied at location `(f, c, c)`.

Fu_eady(i, j, k, grid, time, U, C, p) = @inbounds (- p.α * ℑxzᶠᵃᶜ(i, j, k, grid, U.w)
                                                   - p.α * (grid.zC[k] + p.H) * ∂xᶠᵃᵃ(i, j, k, grid, ℑxᶜᵃᵃ, U.u))

# The $y$-momentum forcing
#
# $ F_v = - α (z + H) ∂ₓv
#
# is applied at location `(c, f, c)`.

Fv_eady(i, j, k, grid, time, U, C, p) = @inbounds -p.α * (grid.zC[k] + p.H) * ∂xᶜᵃᵃ(i, j, k, grid, ℑxᶠᵃᵃ, U.v)

# The $z$-momentum forcing
#
# $ F_w = - α (z + H) ∂ₓw
#
# is applied at location `(c, c, f)`.

Fw_eady(i, j, k, grid, time, U, C, p) = @inbounds -p.α * (grid.zF[k] + p.H) * ∂xᶜᵃᵃ(i, j, k, grid, ℑxᶠᵃᵃ, U.w)

# The buoyancy forcing
#
# $ F_b = - α (z + H) ∂ₓb + α f v $
#
# is applied at location `(c, c, c)`.

Fb_eady(i, j, k, grid, time, U, C, p) = @inbounds (- p.α * (grid.zC[k] + p.H) * ∂xᶜᵃᵃ(i, j, k, grid, ℑxᶠᵃᵃ, C.b)
                                                   + p.f * p.α * ℑyᵃᶜᵃ(i, j, k, grid, U.v))

# # Turbulence closures
#
# We use a horizontal biharmonic diffusivity and a Laplacian vertical diffusivity
# to dissipate energy in the Eady problem.
# Two use both of these closures at the same time, we set the keyword argument
# `closure` a tuple of two closures. Note that the "2D Leith" parameterization may
# also be a sensible choice to pair with a Laplacian vertical diffusivity for this problem.

closure = (ConstantAnisotropicDiffusivity(νh=0, κh=0, νv=κᵥ, κv=κᵥ),
           AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h))
           #TwoDimensionalLeith())

# Form a prefix from chosen resolution, boundary condition, and closure name

output_filename_prefix = string("eady_turb_Nh", Nh, "_Nz", Nz)

# # Model instantiation
#
# Our use of biharmonic diffusivity means we must instantiate the model grid
# with halos of size `(2, 2, 2)` in `(x, y, z)`.

model = Model( grid = RegularCartesianGrid(size=(Nh, Nh, Nz), halo=(2, 2, 2),
                                           x=(-Lh/2, Lh/2), y=(-Lh/2, Lh/2), z=(-Lz, 0)),
       architecture = CPU(),
           coriolis = FPlane(f=f),
           buoyancy = BuoyancyTracer(), tracers = :b,
            forcing = ModelForcing(u=Fu_eady, v=Fv_eady, w=Fw_eady, b=Fb_eady),
            closure = closure,
boundary_conditions = BoundaryConditions(u=ubcs, v=vbcs, b=bbcs),
# "parameters" is a NamedTuple of user-defined parameters that can be used in boundary condition and forcing functions.
         parameters = merge(bc_parameters, forcing_parameters))

# # Initial conditions
#
# For initial conditions we impose a linear stratifificaiton with some
# random noise.

## A noise function, damped at the boundaries
Ξ(z) = rand() * z/Lz * (z/Lz + 1)

## Buoyancy: linear stratification plus noise
b₀(x, y, z) = N² * z + 1e-2 * Ξ(z) * (N² * Lz + α * f * Lh)
u₀(x, y, z) = 1e-2 * α * Lz

set!(model, u=u₀, v=u₀, b=b₀)

# # Diagnostics and output
#
# Diagnostics that return the maximum absolute value of `u, v, w` by calling
# `umax(model), vmax(model), wmax(model)`:

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

# Set up output. Here we output the velocity and buoyancy fields at intervals of one day.
fields_to_output = merge(model.velocities, (b=model.tracers.b,))
output_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output);
                                 interval=day, prefix=output_filename_prefix,
                                 force=true, max_filesize=10GiB)

# # The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the CFL close to a
# desired value.

wizard = TimeStepWizard(cfl=0.05, Δt=20.0, max_change=1.1, max_Δt=min(1/10f, σᵇ/10))

# # Vertical vorticity and divergence for plotting purposes
#
# We also create objects for computing the vertical vorticity and divergence
# for plotting purposes.

u, v, w = model.velocities
ζ = Field(Face, Face, Cell, model.architecture, model.grid)
δ = Field(Cell, Cell, Cell, model.architecture, model.grid)

vertical_vorticity = Computation(∂x(v) - ∂y(u), ζ)
        divergence = Computation(-∂z(w), δ)

ζmax = FieldMaximum(abs, ζ)
δmax = FieldMaximum(abs, δ)

# # Plot setup
#
# Set `makeplot = true` to make plots as the model runs.

makeplot = false

using PyPlot, PyCall

GridSpec = pyimport("matplotlib.gridspec").GridSpec

fig = figure(figsize=(12, 8))

gs = GridSpec(2, 2, height_ratios=[2, 1])

axs = ntuple(4) do i
    fig.add_subplot(get(gs, i-1))
end

function makeplot!(axs, model)
    nx, ny, nz = size(model.grid)

    xC_xy = repeat(reshape(model.grid.xC, nx, 1), 1, ny)
    xF_xy = repeat(reshape(model.grid.xF[1:end-1], nx, 1), 1, ny)

    yC_xy = repeat(reshape(model.grid.yC, 1, ny), nx, 1)
    yF_xy = repeat(reshape(model.grid.yF[1:end-1], 1, ny), nx, 1)

    xC_xz = repeat(reshape(model.grid.xC, nx, 1), 1, nz)
    xF_xz = repeat(reshape(model.grid.xF[1:end-1], nx, 1), 1, nz)

    zC_xz = repeat(reshape(model.grid.zC, 1, nz), nx, 1)
    zF_xz = repeat(reshape(model.grid.zF[1:end-1], 1, nz), nx, 1)

    compute!(vertical_vorticity)
    compute!(divergence)

    @printf("\nmax ζ/f: %.2e, max δ/f: %.2e\n\n", ζmax()/f, δmax()/f)

    sca(axs[1]); cla()
    pcolormesh(xF_xy/1e3, yF_xy/1e3, Array(interior(ζ)[:, :, Nz]))
    title("Surface vertical vorticity (1/s)"); xlabel("\$ x \$ (km)"); ylabel("\$ y \$ (km)")
    axs[1].set_aspect(1)

    sca(axs[2]); cla()
    pcolormesh(xC_xy/1e3, yC_xy/1e3, Array(interior(δ)[:, :, Nz]))
    title("Surface divergence (1/s)"); xlabel("\$ x \$ (km)"); ylabel("\$ y \$ (km)")
    axs[2].set_aspect(1)

    sca(axs[3]); cla()
    pcolormesh(xF_xz/1e3, zC_xz, Array(interior(ζ)[:, Int(Nh/2), :]))
    title("Vertical vorticity (1/s)"); xlabel("\$ x \$ (km)"); ylabel("\$ z \$ (m)")

    sca(axs[4]); cla()
    pcolormesh(xC_xz/1e3, zF_xz, Array(interior(w)[:, Int(Nh/2), :]))
    title("Vertical velocity (m/s)"); xlabel("\$ x \$ (km)"); ylabel("\$ z \$ (m)")

    tight_layout()
    pause(0.1)

    return nothing
end

# # Plot setup
#
# This time-stepping loop runs until end_time is reached. It prints a progress statement
# every 100 iterations.

while model.clock.time < end_time

    ## Update the time step associated with `wizard`.
    update_Δt!(wizard, model)

    ## Time step the model forward
    walltime = Base.@elapsed time_step!(model, 10, wizard.Δt)

    ## Print a progress message
    @printf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
            model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
            umax(), vmax(), wmax(), prettytime(walltime))

    if model.clock.iteration % 100 == 0 && makeplot
        makeplot!(axs, model)
    end
end

# Make a plot at the end

makeplot!(axs, model)
gcf()
