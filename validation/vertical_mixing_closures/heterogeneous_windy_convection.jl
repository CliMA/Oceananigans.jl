using Printf
using Statistics
using GLMakie

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom

import Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities

Nx = 50
Ny = 1

const Lx = 500kilometers
const Ly = Lx
const Lz = 1000

# Stretched vertical grid
γ = 1.01
Δz₀ = 16
h₀ = 128
z = [-Δz₀ * k for k = 0:ceil(h₀ / Δz₀)]
while z[end] > -Lz
    push!(z, z[end] - (z[end-1] - z[end])^γ)
end
z = reverse(z)
Nz = length(z) - 1

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       halo = (4, 4, 4),
                       x = (-Lx, 0),
                       y = (0, Ly),
                       z = z,
                       topology=(Bounded, Periodic, Bounded))

z_bottom(x, y) = - 2 * abs(x) * Lz / Lx
#grid = ImmersedBoundaryGrid(grid, PartialCellBottom(z_bottom, minimum_fractional_cell_height=0.2))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(z_bottom))

@show grid
@inline Qᵇ(x, y, t) = 0.0 #1e-7
@inline Qᵘ(x, y, t) = 0.0
@inline Qᵛ(x, y, t, p) = + 1e-4 * exp(-x^2 / (2 * p.δx^2))

b_top_bc = FluxBoundaryCondition(Qᵇ)
u_top_bc = FluxBoundaryCondition(Qᵘ)
v_top_bc = FluxBoundaryCondition(Qᵛ, parameters=(; δx=200kilometers))

b_bcs = FieldBoundaryConditions(top=b_top_bc)
u_bcs = FieldBoundaryConditions(top=u_top_bc)
v_bcs = FieldBoundaryConditions(top=v_top_bc)

vertical_mixing = CATKEVerticalDiffusivity(; minimum_turbulent_kinetic_energy=1e-6)
#vertical_mixing = RiBasedVerticalDiffusivity()
#
horizontal_viscosity = HorizontalScalarDiffusivity(ν=1e4)

@show vertical_mixing
#closure = (vertical_mixing, horizontal_viscosity)
closure = vertical_mixing

filename = "heterogeneous_cooling.jld2"

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    coriolis = FPlane(latitude=+33),
                                    tracers = (:b, :e),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs, v=v_bcs),
                                    buoyancy = BuoyancyTracer())

N²ᵢ = 1e-5
bᵢ(x, y, z) = N²ᵢ * z
set!(model, b=bᵢ, e=1e-6)

simulation = Simulation(model, Δt=10minute, stop_iteration=1000)

κᶜ = if model.closure isa Tuple
    model.diffusivity_fields[1].κᶜ
else
    model.diffusivity_fields.κᶜ
end

b = model.tracers.b
N² = ∂z(b)
outputs = (; model.velocities..., model.tracers..., κᶜ=κᶜ, N²=N²)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs;
                                                      filename,
                                                      #schedule = TimeInterval(1hour),
                                                      schedule = IterationInterval(10),
                                                      overwrite_existing = true)

function progress(sim)
    u, v, w = sim.model.velocities
    e = sim.model.tracers.e


    msg = @sprintf("Iter: %d, t: %s, max|u|: (%6.2e, %6.2e, %6.2e) m s⁻¹", 
                   iteration(sim), prettytime(sim),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))

    msg *= @sprintf(", max(e): %6.2e m² s⁻²", maximum(e))
    msg *= @sprintf(", max(κᶜ): %6.2e m² s⁻¹", maximum(κᶜ))

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
κ_ts = FieldTimeSeries(filename, "κᶜ")
N_ts = FieldTimeSeries(filename, "N²")
Nt = length(b_ts.times)

for ψ in (b_ts, e_ts, u_ts, v_ts, w_ts, κ_ts) #, N_ts)
    ψp = parent(ψ)
    ψp[ψp .== 0] .= NaN
end

fig = Figure(resolution=(1600, 800))

ax_vxz = Axis(fig[1, 1], title="v(x, z) - <v(x, z)>")
ax_wxz = Axis(fig[1, 2], title="w(x, z)")
ax_Nxz = Axis(fig[1, 3], title="N²(x, z)")
ax_exz = Axis(fig[1, 4], title="e(x, z)")
ax_κxz = Axis(fig[1, 5], title="κ(x, z)")

ax_bz = Axis(fig[2, 1], title="b(z)", xlabel="z")
ax_uz = Axis(fig[2, 2], title="u(z)", ylabel="z")
ax_vz = Axis(fig[2, 3], title="v(z)", ylabel="z")
ax_ez = Axis(fig[2, 4], title="e(z)", ylabel="z")
ax_κz = Axis(fig[2, 5], title="κ(z)", ylabel="z")

slider = Slider(fig[3, :], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Two-dimensional channel at t = ", prettytime(b_ts.times[$n]))
Label(fig[0, :], title, fontsize=24)

b_xz = @lift interior(b_ts[$n], :, 1, :)
e_xz = @lift interior(e_ts[$n], :, 1, :)

u_xz = @lift begin
    u = interior(u_ts[$n], :, 1, :)
    u .- mean(filter(!isnan, u))
end

v_xz = @lift begin
    v = interior(v_ts[$n], :, 1, :)
    v .- mean(filter(!isnan, v))
end

v_xz = @lift interior(v_ts[$n], :, 1, :)
w_xz = @lift interior(w_ts[$n], :, 1, :)
w_xz = @lift interior(w_ts[$n], :, 1, :)
N_xz = @lift interior(N_ts[$n], :, 1, :)
κ_xz = @lift interior(κ_ts[$n], :, 1, :)

Nx, Ny, Nz = size(grid)

b_z1 = @lift interior(b_ts[$n], 16, 1, :)
b_z2 = @lift interior(b_ts[$n], 32, 1, :)
b_z3 = @lift interior(b_ts[$n], 8,  1, :)

e_z1 = @lift interior(e_ts[$n], 16, 1, :)
e_z2 = @lift interior(e_ts[$n], 32, 1, :)
e_z3 = @lift interior(e_ts[$n], 8,  1, :)

κ_z1 = @lift interior(κ_ts[$n], 16, 1, :)
κ_z2 = @lift interior(κ_ts[$n], 32, 1, :)
κ_z3 = @lift interior(κ_ts[$n], 8,  1, :)

u_z1 = @lift interior(u_ts[$n], 16, 1, :)
u_z2 = @lift interior(u_ts[$n], 32, 1, :)
u_z3 = @lift interior(u_ts[$n], 8,  1, :)
                                       
v_z1 = @lift interior(v_ts[$n], 16, 1, :)
v_z2 = @lift interior(v_ts[$n], 32, 1, :)
v_z3 = @lift interior(v_ts[$n], 8,  1, :)

x, y, z = nodes(b_ts)
xκ, yκ, zκ = nodes(κ_ts)

elim = 1e-4
ulim = 0.2
vlim = 1e-4
wlim = 1e-7
κlim = 1e-3 # 1e1

heatmap!(ax_exz, x, z, e_xz, colormap=:solar, colorrange=(0, elim), nan_color=:gray)
contour!(ax_exz, x, z, b_xz, levels=15, color=:black)

heatmap!(ax_κxz, x, zκ, κ_xz, colormap=:thermal, colorrange=(0, κlim), nan_color=:gray)
contour!(ax_κxz, x, z, b_xz, levels=15, color=:black)

# heatmap!(ax_uxz, x, z, u_xz, colormap=:balance, colorrange=(-ulim, ulim), nan_color=:gray)
# contour!(ax_uxz, x, z, b_xz, levels=15, color=:black)

heatmap!(ax_vxz, x, z, v_xz, colormap=:balance, colorrange=(-vlim, vlim), nan_color=:gray)
contour!(ax_vxz, x, z, b_xz, levels=15, color=:black)

heatmap!(ax_wxz, x, z, w_xz, colormap=:balance, colorrange=(-wlim, wlim), nan_color=:gray)
contour!(ax_wxz, x, z, b_xz, levels=15, color=:black)

heatmap!(ax_Nxz, x, z, N_xz, colormap=:thermal, colorrange=(1e-6, 2e-5), nan_color=:gray)
contour!(ax_Nxz, x, z, b_xz, levels=15, color=:black)

lines!(ax_bz, b_z1, z)
lines!(ax_bz, b_z2, z)
lines!(ax_bz, b_z3, z)

lines!(ax_ez, e_z1, z)
lines!(ax_ez, e_z2, z)
lines!(ax_ez, e_z3, z)

lines!(ax_κz, κ_z1, zκ)
lines!(ax_κz, κ_z2, zκ)
lines!(ax_κz, κ_z3, zκ)

lines!(ax_uz, u_z1, z)
lines!(ax_uz, u_z2, z)
lines!(ax_uz, u_z3, z)

lines!(ax_vz, v_z1, z)
lines!(ax_vz, v_z2, z)
lines!(ax_vz, v_z3, z)

xlims!(ax_ez, -elim/10, 2elim)
xlims!(ax_uz, -2ulim, 2ulim)
xlims!(ax_vz, -2vlim, 2vlim)
xlims!(ax_κz, -κlim/10, 2κlim)

ylims!(ax_bz, -1020, 20)
ylims!(ax_uz, -1020, 20)
ylims!(ax_vz, -1020, 20)
ylims!(ax_ez, -1020, 20)
ylims!(ax_κz, -1020, 20)

display(fig)

#=
record(fig, filename[1:end-5] * ".mp4", 1:Nt, framerate=24) do nn
    @info "Plotting frame $nn of $Nt..."
    n[] = nn
end
=#

