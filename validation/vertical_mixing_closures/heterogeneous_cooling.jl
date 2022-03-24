using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

Nx = 1
Ny = 2
Nz = 64

Lx = Ly = 100kilometers
Lz = 400

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo=(3, 3, 3),
                       x=(0, Lx), y=(-Ly/2, Ly/2), z=(-Lz, 0),
                       topology=(Periodic, Bounded, Bounded))

Qᵇ(x, y, t) = 2e-8 * y / Ly
Qᵘ(x, y, t) = 0.0 #-1e-4
b_top_bc = FluxBoundaryCondition(Qᵇ)
u_top_bc = FluxBoundaryCondition(Qᵘ)
b_bcs = FieldBoundaryConditions(top=b_top_bc)
u_bcs = FieldBoundaryConditions(top=u_top_bc)

Δh = Ly / Ny
κ₄ = Δh^4 / 30days
biharmonic_closure = HorizontalScalarBiharmonicDiffusivity(ν=κ₄, κ=κ₄)
#boundary_layer_closure = CATKEVerticalDiffusivity()
boundary_layer_closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1)

closure = (boundary_layer_closure, biharmonic_closure)

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    tracers = (:b, :e),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs),
                                    buoyancy = BuoyancyTracer())

N² = 1e-5
bᵢ(x, y, z) = N² * z # + 1e-8 * rand()
set!(model, b=bᵢ)

simulation = Simulation(model, Δt=2minutes, stop_iteration=1000)

#=
slice_indices = (
    west   = (1,  :, :),
    east   = (Nx, :, :),
    south  = (:, 1,  :),
    north  = (:, Ny, :),
    bottom = (:, :, 1),
    top    = (:, :, Nz),
)

for side in keys(indices)
    indices = slice_indices[side]

    simulation.output_writers[:side] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                         schedule = TimeInterval(10minutes),
                         prefix = "heterogeneous_cooling_" * side,
                         indices,
                         force = true)
end
=#

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                     schedule = TimeInterval(4hours),
                     prefix = "heterogeneous_cooling",
                     force = true)

function progress(sim)
    u, v, w = sim.model.velocities

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|u|: (%6.2e, %6.2e, %6.2e) m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))

    @info msg
    
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

run!(simulation)

using GLMakie

fig = Figure(resolution=(800, 1200))

b = model.tracers.b
b_xy = interior(b, :, :, Nz)
b_xz = interior(b, :, 1, :)
b_yz = interior(b, 1, :, :)

x, y, z = nodes(b)

u = model.velocities.u
v = model.velocities.v
shear_op = @at (Center, Center, Center) sqrt(∂z(u)^2 + ∂z(v)^2)
shear = compute!(Field(shear_op))
shear_yz = interior(shear, 1, :, :)

ax_xy = Axis(fig[1, 1], aspect=1)
ax_yz = Axis(fig[2, 1], aspect=2)

heatmap!(ax_xy, x, y, b_xy)
heatmap!(ax_yz, y, z, shear_yz)
contour!(ax_yz, y, z, b_yz, levels=15)

#=
ax = Axis3(fig[1, 1], aspect=:data)

b = model.tracers.b
bxy = interior(b, :, :, Nz)
bxz = interior(b, :, 1, :)
byz = interior(b, 1, :, :)

x, y, z = nodes(b)

x_xz = repeat(x, 1, Nz)
y_xz = 0.995 * Ly * ones(Nx, Nz)
z_xz = repeat(reshape(z, 1, Nz), Nx, 1)

x_yz = 0.995 * Lx * ones(Ny, Nz)
y_yz = repeat(y, 1, Nz)
z_yz = repeat(reshape(z, 1, Nz), Ny, 1)

x_xy = x
y_xy = y
z_xy = -0.001 * Lz * ones(Nx, Ny)

surface!(ax, x_xz, y_xz, z_xz, color=bxz)
surface!(ax, x_yz, y_yz, z_yz, color=byz)
surface!(ax, x_xy, y_xy, z_xy, color=bxy)
=#

display(fig)
