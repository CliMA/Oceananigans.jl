using Printf
using GLMakie
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

Nx = 128
Ny = 128
Nz = 32

Lx = Ly = 100kilometers
Lz = 200

grid = RectilinearGrid(size=(Nx, Ny, Nz), halo=(3, 3, 3),
                       x=(0, Lx), y=(0, Ly), z=(-Lz, 0),
                       topology=(Periodic, Bounded, Bounded))

@show grid

Qᵇ(x, y, t) = 2e-8 * y / Ly
Qᵘ(x, y, t) = -1e-5
b_top_bc = FluxBoundaryCondition(Qᵇ)
u_top_bc = FluxBoundaryCondition(Qᵘ)
b_bcs = FieldBoundaryConditions(top=b_top_bc)
u_bcs = FieldBoundaryConditions(top=u_top_bc)

Δh = Ly / Ny
κ₄ = Δh^4 / 1day
κ₂ = Δh^2 / 10days

horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν=κ₄, κ=κ₄)
#horizontal_closure = HorizontalScalarDiffusivity(ν=κ₂, κ=κ₂)
boundary_layer_closure = CATKEVerticalDiffusivity()
#boundary_layer_closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1)

closure = (boundary_layer_closure, horizontal_closure)
#closure = boundary_layer_closure

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    tracers = (:b, :e),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs),
                                    buoyancy = BuoyancyTracer())

N² = 1e-5
h = Lz / 3
bᵢ(x, y, z) = N² * z #N² * h * exp(z / h) # + 1e-8 * rand()
set!(model, b=bᵢ)

simulation = Simulation(model, Δt=1minutes, stop_iteration=10000)

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
                     schedule = TimeInterval(5minutes),
                     prefix = "heterogeneous_cooling",
                     force = true)

function progress(sim)
    u, v, w = sim.model.velocities
    e = sim.model.tracers.e

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|u|: (%6.2e, %6.2e, %6.2e) m s⁻¹", 
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))

    msg *= @sprintf(", max(e): %6.2e m² s⁻²", maximum(abs, e))

    @info msg
    
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

run!(simulation)

filepath = "heterogeneous_cooling.jld2"
b_ts = FieldTimeSeries(filepath, "b")
e_ts = FieldTimeSeries(filepath, "e")
u_ts = FieldTimeSeries(filepath, "u")
v_ts = FieldTimeSeries(filepath, "v")
Nt = length(b_ts.times)

fig = Figure(resolution=(1600, 1200))

ax_bxy = Axis(fig[1, 1], aspect=1)
ax_exy = Axis(fig[1, 2], aspect=1)
ax_syz = Axis(fig[2, 1], aspect=2)
ax_eyz = Axis(fig[2, 2], aspect=2)
slider = Slider(fig[3, :], range=1:Nt, startvalue=1)
n = slider.value

b_xy = @lift interior(b_ts[$n], :, :, Nz)
b_xz = @lift interior(b_ts[$n], :, 1, :)
b_yz = @lift interior(b_ts[$n], 1, :, :)

e_xy = @lift interior(e_ts[$n], :, :, Nz)
e_xz = @lift interior(e_ts[$n], :, 1, :)
e_yz = @lift interior(e_ts[$n], 1, :, :)

x, y, z = nodes(b_ts)

shear_yz = @lift begin
    u = u_ts[$n]
    v = v_ts[$n]
    shear_u = compute!(Field(∂z(u)^2))
    shear_v = compute!(Field(∂z(v)^2))
    shear_op = @at (Center, Center, Center) sqrt(shear_u + shear_v)
    shear = compute!(Field(shear_op))
    interior(shear, 1, :, :)
end

heatmap!(ax_bxy, x, y, b_xy)
heatmap!(ax_exy, x, y, e_xy)

heatmap!(ax_syz, y, z, shear_yz)
contour!(ax_syz, y, z, b_yz, levels=15)

heatmap!(ax_eyz, y, z, e_yz)
contour!(ax_eyz, y, z, b_yz, levels=15)

display(fig)

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
