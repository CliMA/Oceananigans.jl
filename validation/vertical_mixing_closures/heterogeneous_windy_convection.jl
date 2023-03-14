using Printf
using GLMakie
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization

Nx = 1
Ny = 64
Nz = 32

const Lx = 100kilometers
const Ly = Lx
const Lz = 256

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology=(Periodic, Bounded, Bounded))

@show grid
@inline Qᵇ(x, y, t) = 2e-8 #* sin(2π * y / Ly)
@inline Qᵘ(x, y, t) = -1e-4 * sin(π * y / Ly)

b_top_bc = FluxBoundaryCondition(Qᵇ)
u_top_bc = FluxBoundaryCondition(Qᵘ)

b_bcs = FieldBoundaryConditions(top=b_top_bc)
u_bcs = FieldBoundaryConditions(top=u_top_bc)

etd = ExplicitTimeDiscretization()
closure = CATKEVerticalDiffusivity()

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    #coriolis = FPlane(f=1e-4),
                                    momentum_advection = WENO(),
                                    #tracer_advection = (b=WENO(), e=CenteredSecondOrder()),
                                    tracer_advection = (b=WENO(), e=WENO()),
                                    #tracer_advection = (b=WENO(), e=nothing),
                                    tracers = (:b, :e),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs),
                                    buoyancy = BuoyancyTracer())

N² = 1e-5
h = Lz / 3
bᵢ(x, y, z) = N² * z
set!(model, b=bᵢ, e=1e-6)

simulation = Simulation(model, Δt=10minute, stop_iteration=400)

filename = "heterogeneous_cooling.jld2"
simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                                                      filename,
                                                      schedule = IterationInterval(1),
                                                      overwrite_existing = true)

function progress(sim)
    u, v, w = sim.model.velocities
    e = sim.model.tracers.e
    κᶜ = sim.model.diffusivity_fields.Kᶜ

    msg = @sprintf("Iter: %d, t: %s, max|u|: (%6.2e, %6.2e, %6.2e) m s⁻¹", 
                   iteration(sim), prettytime(sim),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))

    msg *= @sprintf(", extrema(e): (%6.2e, %6.2e) m² s⁻²", minimum(e), maximum(e))
    msg *= @sprintf(", extrema(κᶜ): (%6.2e, %6.2e) m² s⁻²", minimum(κᶜ), maximum(κᶜ))

    @info msg
    
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

run!(simulation)

b_ts = FieldTimeSeries(filename, "b")
e_ts = FieldTimeSeries(filename, "e")
u_ts = FieldTimeSeries(filename, "u")
v_ts = FieldTimeSeries(filename, "v")
w_ts = FieldTimeSeries(filename, "w")
Nt = length(b_ts.times)

fig = Figure(resolution=(1600, 1200))

ax_bxy = Axis(fig[1, 1])
ax_uxy = Axis(fig[1, 2])
ax_exy = Axis(fig[1, 3])
ax_vyz = Axis(fig[2, 1])
ax_wyz = Axis(fig[2, 2])
ax_eyz = Axis(fig[2, 3])

ax_ey = Axis(fig[3, 1:2])
ax_ez = Axis(fig[3, 3])

slider = Slider(fig[4, :], range=1:Nt, startvalue=1)
n = slider.value

b_xy = @lift interior(b_ts[$n], :, :, Nz)
b_xz = @lift interior(b_ts[$n], :, 1, :)
b_yz = @lift interior(b_ts[$n], 1, :, :)

e_xy = @lift interior(e_ts[$n], :, :, Nz)
e_xz = @lift interior(e_ts[$n], :, 1, :)
e_yz = @lift interior(e_ts[$n], 1, :, :)

e_y1 = @lift interior(e_ts[$n], 1, :, 32)
e_y2 = @lift interior(e_ts[$n], 1, :, 30)
e_y3 = @lift interior(e_ts[$n], 1, :, 28)
e_z1 = @lift interior(e_ts[$n], 1, 16, :)
e_z2 = @lift interior(e_ts[$n], 1, 32, :)
e_z3 = @lift interior(e_ts[$n], 1, 8, :)

u_xy = @lift interior(u_ts[$n], :, :, Nz)
u_xz = @lift interior(u_ts[$n], :, 1, :)
u_yz = @lift interior(u_ts[$n], 1, :, :)

v_xy = @lift interior(v_ts[$n], :, :, Nz)
v_xz = @lift interior(v_ts[$n], :, 1, :)
v_yz = @lift interior(v_ts[$n], 1, :, :)

w_xy = @lift interior(w_ts[$n], :, :, Nz)
w_xz = @lift interior(w_ts[$n], :, 1, :)
w_yz = @lift interior(w_ts[$n], 1, :, :)

x, y, z = nodes(b_ts)

heatmap!(ax_bxy, x, y, b_xy)
heatmap!(ax_uxy, x, y, u_xy)
heatmap!(ax_exy, x, y, e_xy)

heatmap!(ax_eyz, y, z, e_yz)
#contour!(ax_eyz, y, z, b_yz, levels=15, linecolor=:black)

heatmap!(ax_vyz, y, z, v_yz)
#contour!(ax_uyz, y, z, b_yz, levels=15, linecolor=:black)

heatmap!(ax_wyz, y, z, w_yz)
#contour!(ax_vyz, y, z, b_yz, levels=15, linecolor=:black)

heatmap!(ax_vyz, y, z, v_yz)
#contour!(ax_vyz, y, z, b_yz, levels=15, linecolor=:black)

lines!(ax_ey, y, e_y1)
lines!(ax_ey, y, e_y2)
lines!(ax_ey, y, e_y3)
lines!(ax_ez, e_z1, z)
lines!(ax_ez, e_z2, z)
lines!(ax_ez, e_z3, z)

ylims!(ax_ey, -1e-5, 4e-4)
xlims!(ax_ez, -1e-5, 4e-4)

display(fig)

