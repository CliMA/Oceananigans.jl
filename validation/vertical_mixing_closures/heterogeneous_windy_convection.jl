using Printf
using Statistics
using GLMakie

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom

import Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities

Nx = 1
Ny = 200

const Lx = 1000kilometers
const Ly = Lx
const Lz = 1000

# Stretched vertical grid
γ = 1.01
Δz₀ = 8
h₀ = 128
z = [-Δz₀ * k for k = 0:ceil(h₀ / Δz₀)]
while z[end] > -Lz
    push!(z, z[end] - (z[end-1] - z[end])^γ)
end
z = reverse(z)
Nz = length(z) - 1

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       halo = (4, 4, 4),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = z,
                       topology=(Periodic, Bounded, Bounded))

z_bottom(x, y) = - Lz * (1 - (2y / Ly)^2)
grid = ImmersedBoundaryGrid(grid, PartialCellBottom(z_bottom, minimum_fractional_cell_height=0.1))

@show grid
@inline Qᵇ(x, y, t) = 1e-7
@inline Qᵘ(x, y, t) = -1e-3 * cos(π * y / Ly)

b_top_bc = FluxBoundaryCondition(Qᵇ)
u_top_bc = FluxBoundaryCondition(Qᵘ)

b_bcs = FieldBoundaryConditions(top=b_top_bc)
u_bcs = FieldBoundaryConditions(top=u_top_bc)

vertical_mixing = CATKEVerticalDiffusivity(; minimum_turbulent_kinetic_energy=1e-9)
#vertical_mixing = RiBasedVerticalDiffusivity()

@show vertical_mixing

Δy = Ly / Ny
ν₄ = Δy^4 / 1hours
hyperviscosity = HorizontalScalarBiharmonicDiffusivity(ν=ν₄)

#closure = vertical_mixing
closure = (vertical_mixing, hyperviscosity)

filename = "heterogeneous_cooling_with_hyperviscosity.jld2"
#filename = "heterogeneous_cooling.jld2"

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    coriolis = FPlane(f=1e-4),
                                    tracers = (:b, :e),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs),
                                    buoyancy = BuoyancyTracer())

N²ᵢ = 1e-5
bᵢ(x, y, z) = N²ᵢ * z
set!(model, b=bᵢ, e=1e-9)

simulation = Simulation(model, Δt=5minute, stop_time=10days)

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
                                                      schedule = TimeInterval(1hour),
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

ax_uyz = Axis(fig[1, 1], title="u(y, z) - <u(y, z)>")
#ax_vyz = Axis(fig[1, 2], title="v(y, z)")
ax_wyz = Axis(fig[1, 2], title="w(y, z)")
ax_Nyz = Axis(fig[1, 3], title="N²(y, z)")
ax_eyz = Axis(fig[1, 4], title="e(y, z)")
ax_κyz = Axis(fig[1, 5], title="κ(y, z)")

ax_bz = Axis(fig[2, 1], title="b(z)", xlabel="z")
ax_uz = Axis(fig[2, 2], title="u(z)", ylabel="z")
ax_vz = Axis(fig[2, 3], title="v(z)", ylabel="z")
ax_ez = Axis(fig[2, 4], title="e(z)", ylabel="z")
ax_κz = Axis(fig[2, 5], title="κ(z)", ylabel="z")

slider = Slider(fig[3, :], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Two-dimensional channel at t = ", prettytime(b_ts.times[$n]))
Label(fig[0, :], title, fontsize=24)

b_yz = @lift interior(b_ts[$n], 1, :, :)
e_yz = @lift interior(e_ts[$n], 1, :, :)

u_yz = @lift begin
    u = interior(u_ts[$n], 1, :, :)
    u .- mean(filter(!isnan, u))
end

v_yz = @lift interior(v_ts[$n], 1, :, :)
w_yz = @lift interior(w_ts[$n], 1, :, :)
w_yz = @lift interior(w_ts[$n], 1, :, :)
N_yz = @lift interior(N_ts[$n], 1, :, :)
κ_yz = @lift interior(κ_ts[$n], 1, :, :)

Nx, Ny, Nz = size(grid)

b_z1 = @lift interior(b_ts[$n], 1, 16, :)
b_z2 = @lift interior(b_ts[$n], 1, 32, :)
b_z3 = @lift interior(b_ts[$n], 1, 8, :)

e_z1 = @lift interior(e_ts[$n], 1, 16, :)
e_z2 = @lift interior(e_ts[$n], 1, 32, :)
e_z3 = @lift interior(e_ts[$n], 1, 8, :)

κ_z1 = @lift interior(κ_ts[$n], 1, 16, :)
κ_z2 = @lift interior(κ_ts[$n], 1, 32, :)
κ_z3 = @lift interior(κ_ts[$n], 1, 8, :)

u_z1 = @lift interior(u_ts[$n], 1, 16, :)
u_z2 = @lift interior(u_ts[$n], 1, 32, :)
u_z3 = @lift interior(u_ts[$n], 1, 8, :)

v_z1 = @lift interior(v_ts[$n], 1, 16, :)
v_z2 = @lift interior(v_ts[$n], 1, 32, :)
v_z3 = @lift interior(v_ts[$n], 1, 8, :)

x, y, z = nodes(b_ts)
xκ, yκ, zκ = nodes(κ_ts)

elim = 1e-4
ulim = 0.2
vlim = 2e-2
wlim = 1e-5
κlim = 1e-3 # 1e1

heatmap!(ax_eyz, y, z, e_yz, colormap=:solar, colorrange=(0, elim), nan_color=:gray)
contour!(ax_eyz, y, z, b_yz, levels=15, color=:black)

heatmap!(ax_κyz, y, zκ, κ_yz, colormap=:thermal, colorrange=(0, κlim), nan_color=:gray)
contour!(ax_κyz, y, z, b_yz, levels=15, color=:black)

heatmap!(ax_uyz, y, z, u_yz, colormap=:balance, colorrange=(-ulim, ulim), nan_color=:gray)
contour!(ax_uyz, y, z, b_yz, levels=15, color=:black)

# heatmap!(ax_vyz, y, z, v_yz, colormap=:balance, colorrange=(-vlim, vlim), nan_color=:gray)
# contour!(ax_vyz, y, z, b_yz, levels=15, color=:black)

heatmap!(ax_wyz, y, z, w_yz, colormap=:balance, colorrange=(-wlim, wlim), nan_color=:gray)
contour!(ax_wyz, y, z, b_yz, levels=15, color=:black)

heatmap!(ax_Nyz, y, z, N_yz, colormap=:thermal, colorrange=(1e-6, 1e-5), nan_color=:gray)
contour!(ax_Nyz, y, z, b_yz, levels=15, color=:black)

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

