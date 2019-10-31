# # Eady turbulence example
#
# In this example, we initialize a random velocity field and observe its viscous,
# turbulent decay in a two-dimensional domain. This example demonstrates:
#
#   * How to use a tuple of turbulence closures
#   * How to use biharmonic diffusivity
#   * How to implement a background flow (a background geostrophic shear)

using Oceananigans, Random, Printf, Oceananigans.AbstractOperations

using Oceananigans: Face, Cell

#####
##### Parameters
#####

# Resolution
Nh = 96                  # horizontal resolution
Nz = 16                  # vertical resolution

# Domain size            
Lh = 1000e3              # [meters] horizontal domain extent
Lz = 1000                # [meters] vertical domain extent
Δh = Lh / Nh             # [meters] horizontal grid spacing for diffusivity calculations
Δz = Lz / Nz             # [meters] vertical grid spacing for diffusivity calculations

# Physical parameters
 f = 1e-4               # [s⁻¹] Coriolis parameter
N² = 1e-6               # [s⁻²] Initial buoyancy gradient 
 α = 1e-2               # [s⁻¹] background shear
 μ = 1/30day            # [s⁻¹] background shear

  τ = 0.25day           # [s] damping time-scale
κ₄ₕ = Δh^4 / τ          # [m⁴ s⁻¹] Biharmonic horizontal diffusivity
 κᵥ = Δz / τ            # [m² s⁻¹] Laplacian vertical diffusivity

end_time = 60day # Simulation end time

# These functions define various physical boundary conditions, 
# and are defined in the file eady_utils.jl

#####
##### Choose boundary conditions and the turbulence closure
#####

bc_parameters = (μ=μ, H=Lz)

@inline τ₁₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * p.H * U.u[i, j, 1]
@inline τ₂₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * p.H * U.v[i, j, 1]

ubcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₁₃_linear_drag))
vbcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₂₃_linear_drag))
bbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 0), 
                               bottom = BoundaryCondition(Value, N² * Lz))

# Get forcing functions and parameters for a linear geostrophic flow ψ = -α y z, where
# α is the geostrophic shear and horizontal buoyancy gradient.
forcing_parameters = (α=α, f=f)

Fu_eady(i, j, k, grid, time, U, C, p) = @inbounds (- p.α * ▶xz_fac(i, j, k, grid, U.w)
                                                   - p.α * (grid.zC[k] + grid.Lz) * ∂x_faa(i, j, k, grid, ▶x_caa, U.u))

# Fv = - α (z + H) ∂ₓv is applied at location (c, f, c).  
Fv_eady(i, j, k, grid, time, U, C, p) = @inbounds -p.α * (grid.zC[k] + grid.Lz) * ∂x_caa(i, j, k, grid, ▶x_faa, U.v)

# Fw = - α (z + H) ∂ₓw is applied at location (c, c, f).  
Fw_eady(i, j, k, grid, time, U, C, p) = @inbounds -p.α * (grid.zF[k] + grid.Lz) * ∂x_caa(i, j, k, grid, ▶x_faa, U.w)

# Fb = - α z ∂ₓb + α f v
Fb_eady(i, j, k, grid, time, U, C, p) = @inbounds (- p.α * (grid.zC[k] + grid.Lz) * ∂x_caa(i, j, k, grid, ▶x_faa, C.b)
                                                   + p.f * p.α * ▶y_aca(i, j, k, grid, U.v))

# Turbulence closures: 
closure = (AnisotropicBiharmonicDiffusivity(νh=κ₄ₕ, κh=κ₄ₕ),
           ConstantAnisotropicDiffusivity(νh=0, κh=0, νv=κᵥ, κv=κᵥ))

# Form a prefix from chosen resolution, boundary condition, and closure name
output_filename_prefix = string("eady_turb_Nh", Nh, "_Nz", Nz)

#####
##### Instantiate the model
#####

# Model instantiation
model = Model( grid = RegularCartesianGrid(size=(Nh, Nh, Nz), halo=(2, 2, 2), x=(-Lh/2, Lh/2), y=(-Lh/2, Lh/2), z=(-Lz, 0)),
       architecture = GPU(),
           coriolis = FPlane(f=f),
           buoyancy = BuoyancyTracer(), tracers = :b,
            forcing = ModelForcing(u=Fu_eady, v=Fv_eady, w=Fw_eady, b=Fb_eady),
            closure = closure,
boundary_conditions = BoundaryConditions(u=ubcs, v=vbcs, b=bbcs),
# "parameters" is a NamedTuple of user-defined parameters that can be used in boundary condition and forcing functions.
         parameters = merge(bc_parameters, forcing_parameters))

#####
##### Set initial conditions
#####

# A noise function, damped at the boundaries
Ξ(z) = rand() * z * (z + model.grid.Lz)

# Buoyancy: linear stratification plus noise
b₀(x, y, z) = N² * z + 1e-9 * Ξ(z) * α * f * model.grid.Ly

# Velocity: noise
u₀(x, y, z) = 1e-9 * Ξ(z) * α * model.grid.Lz

set!(model, u=u₀, v=u₀, b=b₀)

#####
##### Set up diagnostics and output
#####

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

# The TimeStepWizard manages the time-step adaptively, keeping the CFL close to a
# desired value.
wizard = TimeStepWizard(cfl=0.05, Δt=20.0, max_change=1.1, max_Δt=10minute)

ζ = Field(Face, Face, Cell, model.architecture, model.grid)
vertical_vorticity = Computation(∂x(v) - ∂y(u), ζ)

#####
##### Time step the model forward
#####

using PyPlot
fig, axs = subplots(ncols=2)

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
            umax(model), vmax(model), wmax(model), prettytime(walltime))

    if model.clock.iteration % 100 == 0
        compute!(vertical_vorticity)

        sca(axs[1]); cla()
        imshow(interior(ζ)[:, :, 1])

        sca(axs[2]); cla()
        imshow(interior(w)[:, 64, :])

        pause(0.1)
    end
end
