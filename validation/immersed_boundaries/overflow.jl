# # Dense water overflow over a coarsely-resolved continental slope
#
# A two-dimensional, non-rotating overflow after Legg, Hallberg and Girton (2006) and Ilıcak et al.
# (2012): dense water rests on a 500 m shelf, spills over the shelf break and descends a continental
# slope into a 2000 m deep basin, held to the slope by a quadratic bottom drag. The vertical grid is
# deliberately coarse -- 100 m levels against a slope that drops 75 m per grid cell -- so the bottom
# is a staircase and the way the immersed boundary represents it matters.
#
# The same configuration is run with `GridFittedBottom`, `PartialCellBottom` and `ShavedCellBottom`
# and compared through the cross-section each bottom leaves open to the flow, the shape of the
# plume, the depth it reaches, and the reference potential energy, which measures the spurious
# diapycnal mixing that a staircased bottom generates.

using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ShavedCellBottom, immersed_cell, immersed_peripheral_node
using Oceananigans.Operators: Vᶜᶜᶜ, Δzᶠᶜᶜ
using Oceananigans.Grids: znode, Center, Face, inactive_node
using Printf
using Statistics: quantile
using CairoMakie

const c = Center()
const f = Face()

#####
##### Configuration
#####

Lx = 200kilometers
Lz = 2000meters

Nx = 400
Nz = 80

Δx = Lx / Nx
Δz = Lz / Nz

shelf_depth = 500meters
shelf_length = 20kilometers
slope_length = 40kilometers

Δb = 0.02        # buoyancy deficit of the dense water, ≈ 2 kg m⁻³
cᴰ = 2.5e-3      # quadratic bottom drag coefficient

stop_time = 12hours
Δt = 5seconds

"""Piecewise-linear bathymetry: a shelf, a continental slope, and a flat abyss."""
function bottom_height(x)
    slope_fraction = (x - shelf_length) / slope_length
    depth = shelf_depth + (Lz - shelf_depth) * clamp(slope_fraction, 0, 1)
    return -depth
end

# A front smoothed over two cells, so that the comparison is not dominated by the overshoots a
# high-order scheme produces at a discontinuity.
initial_buoyancy(x, z) = -Δb/2 * (1 - tanh((x - shelf_length) / 2Δx))

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  halo = (5, 5),
                                  x = (0, Lx),
                                  z = (-Lz, 0),
                                  topology = (Bounded, Flat, Bounded))

immersed_boundaries = ("fitted"  => GridFittedBottom(bottom_height),
                       "partial" => PartialCellBottom(bottom_height, minimum_fractional_cell_height=0.2),
                       "shaved"  => ShavedCellBottom(bottom_height, minimum_fractional_cell_height=0.2))

@inline quadratic_drag(u, cᴰ) = - cᴰ * u * abs(u)

@inline bottom_drag(i, j, grid, clock, fields, cᴰ) = @inbounds quadratic_drag(fields.u[i, j, 1], cᴰ)
@inline immersed_bottom_drag(i, j, k, grid, clock, fields, cᴰ) = @inbounds quadratic_drag(fields.u[i, j, k], cᴰ)

#####
##### Diagnostics
#####

"""
    reference_potential_energy(model)

Return the potential energy of the adiabatic rearrangement of the buoyancy field into a resting,
stably stratified state. Only irreversible mixing can change it, so its drift measures the spurious
diapycnal mixing of the run. Parcels are matched to the volumes they would occupy exactly, which
matters here because the three bottoms give the bottom cells different volumes.
"""
function reference_potential_energy(model)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    b = Array(interior(model.tracers.b))

    zs = Float64[]
    Vs = Float64[]
    bs = Float64[]

    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        immersed_cell(i, j, k, grid) && continue
        push!(zs, znode(i, j, k, grid, c, c, c))
        push!(Vs, Vᶜᶜᶜ(i, j, k, grid))
        push!(bs, b[i, j, k])
    end

    slots = sortperm(zs)          # containers, deepest first
    parcels = sortperm(bs)        # water, densest first

    energy = 0.0
    s = p = 1
    slot_volume = Vs[slots[1]]
    parcel_volume = Vs[parcels[1]]

    while s ≤ length(slots) && p ≤ length(parcels)
        δV = min(slot_volume, parcel_volume)
        energy -= bs[parcels[p]] * zs[slots[s]] * δV

        slot_volume -= δV
        parcel_volume -= δV

        if slot_volume ≤ 0
            s += 1
            s ≤ length(slots) && (slot_volume = Vs[slots[s]])
        end

        if parcel_volume ≤ 0
            p += 1
            p ≤ length(parcels) && (parcel_volume = Vs[parcels[p]])
        end
    end

    return energy
end

"""
Return the depth of the center of mass of the dense water, and the total buoyancy anomaly, which is
conserved and so serves as a check on the run.
"""
function plume_diagnostics(model)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    b = Array(interior(model.tracers.b))

    Σb = 0.0
    Σbz = 0.0

    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        immersed_cell(i, j, k, grid) && continue
        V = Vᶜᶜᶜ(i, j, k, grid)
        Σb += b[i, j, k] * V
        Σbz += b[i, j, k] * znode(i, j, k, grid, c, c, c) * V
    end

    return -Σbz / Σb, Σb
end

"""Fraction of each zonal face that the bottom leaves open to the flow."""
function open_face_fraction(grid)
    fraction = fill(NaN, Nx+1, Nz)

    for i in 1:Nx+1, k in 1:Nz
        west_dry = i > 1 && immersed_cell(i-1, 1, k, grid)
        east_dry = i ≤ Nx && immersed_cell(i, 1, k, grid)
        fraction[i, k] = (west_dry || east_dry) ? 0.0 : Δzᶠᶜᶜ(i, 1, k, grid) / Δz
    end

    return fraction
end

#####
##### Run
#####

function run_overflow(name, immersed_boundary)
    grid = ImmersedBoundaryGrid(underlying_grid, immersed_boundary)

    drag = FluxBoundaryCondition(bottom_drag, discrete_form=true, parameters=cᴰ)
    immersed_drag = FluxBoundaryCondition(immersed_bottom_drag, discrete_form=true, parameters=cᴰ)
    u_boundary_conditions = FieldBoundaryConditions(bottom = drag,
                                                    immersed = ImmersedBoundaryCondition(bottom = immersed_drag))

    model = HydrostaticFreeSurfaceModel(grid;
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        closure = ScalarDiffusivity(ν=1e-2, κ=0),
                                        boundary_conditions = (; u = u_boundary_conditions))

    set!(model, b = initial_buoyancy)

    simulation = Simulation(model; Δt, stop_time)

    # The spanwise vorticity, at (Face, Center, Face), which is where a staircased bottom betrays
    # itself: every step sheds vorticity of its own.
    u, v, w = model.velocities
    vorticity = Field(∂z(u) - ∂x(w))

    times = Float64[]
    energies = Float64[]
    depths = Float64[]
    buoyancies = Float64[]
    buoyancy_snapshots = Matrix{Float32}[]
    vorticity_snapshots = Matrix{Float32}[]

    function record!(sim)
        depth, Σb = plume_diagnostics(sim.model)
        compute!(vorticity)
        push!(times, time(sim))
        push!(energies, reference_potential_energy(sim.model))
        push!(depths, depth)
        push!(buoyancies, Σb)
        push!(buoyancy_snapshots, Float32.(interior(sim.model.tracers.b, :, 1, :)))
        push!(vorticity_snapshots, Float32.(interior(vorticity, :, 1, :)))
        return nothing
    end

    record!(simulation)
    add_callback!(simulation, record!, TimeInterval(5minutes))

    progress(sim) = @info @sprintf("%s: %s, max|u| = %.3f m s⁻¹",
                                   name, prettytime(sim), maximum(abs, sim.model.velocities.u))

    add_callback!(simulation, progress, TimeInterval(2hours))

    @info "Running the $name overflow..."
    run!(simulation)

    return (; name, grid,
            face_fraction = open_face_fraction(grid),
            times, energies, depths, buoyancies,
            buoyancy_snapshots, vorticity_snapshots)
end

results = [run_overflow(name, ib) for (name, ib) in immersed_boundaries]

#####
##### Figure and animation
#####

xᶜ = xnodes(underlying_grid, c) ./ 1e3
xᶠ = xnodes(underlying_grid, f) ./ 1e3
zᶜ = znodes(underlying_grid, c)

xb = range(0, Lx, length=2000)
zb = bottom_height.(xb)

xmax = 90  # km: the shelf, the slope and the start of the abyss

frames = 1:length(first(results).times)
frame = Observable(last(frames))

masks = [[immersed_cell(i, 1, k, result.grid) for i in 1:Nx, k in 1:Nz] for result in results]

# The topography only: the free surface is a legitimate place to hold vorticity.
vorticity_masks = [[immersed_peripheral_node(i, 1, k, result.grid, f, c, f) |
                    inactive_node(i, 1, k, result.grid, f, c, f) for i in 1:Nx+1, k in 1:Nz+1]
                   for result in results]

"""A snapshot at frame `n`, with the cells the boundary blanks out set to `NaN`."""
function masked_snapshot(snapshots, mask, n)
    field = copy(snapshots[n])
    field[mask] .= NaN
    return field
end

fig = Figure(size=(1150, 1300))

for (n, result) in enumerate(results)
    ax = Axis(fig[1, n], title = "$(result.name) bottom", xlabel = "x (km)",
              ylabel = n == 1 ? "z (m)" : "")

    heatmap!(ax, xᶠ, zᶜ, result.face_fraction, colormap = :solar, colorrange = (0, 1))
    lines!(ax, xb ./ 1e3, zb, color = :black, linewidth = 2)

    xlims!(ax, 15, 65)
    ylims!(ax, -Lz, -400)
    n > 1 && hideydecorations!(ax, grid=false)
end

Colorbar(fig[1, 4], colormap = :solar, limits = (0, 1), label = "open fraction of Δzᶠᶜᶜ")

for (n, result) in enumerate(results)
    title = @lift string(result.name, " bottom, t = ", prettytime(result.times[$frame]))

    ax = Axis(fig[n+1, 1:3], title = title,
              xlabel = n == length(results) ? "x (km)" : "",
              ylabel = "z (m)")

    buoyancy = @lift masked_snapshot(result.buoyancy_snapshots, masks[n], $frame)

    heatmap!(ax, xᶜ, zᶜ, buoyancy, colormap = :dense, colorrange = (-Δb, 0), nan_color = :gray80)
    lines!(ax, xb ./ 1e3, zb, color = :black, linewidth = 2)
    xlims!(ax, 0, xmax)
    ylims!(ax, -Lz, 0)
    n < length(results) && hidexdecorations!(ax, grid=false)
end

Colorbar(fig[2:4, 4], colormap = :dense, limits = (-Δb, 0), label = "b (m s⁻²)")

ax_depth = Axis(fig[5, 1:3], xlabel = "time (hours)", ylabel = "depth (m)",
                title = "Depth of the center of mass of the dense water")

ax_energy = Axis(fig[6, 1:3], xlabel = "time (hours)", ylabel = "ΔRPE / RPE(0)",
                 title = "Reference potential energy drift (κ = 0, so all of the mixing is numerical)")

for result in results
    hours = result.times ./ 3600
    drift = (result.energies .- result.energies[1]) ./ abs(result.energies[1])

    depth_line = lines!(ax_depth, hours, result.depths, label = result.name, linewidth = 2)
    energy_line = lines!(ax_energy, hours, drift, label = result.name, linewidth = 2)

    scatter!(ax_depth, @lift(Point2f(hours[$frame], result.depths[$frame])),
             color = depth_line.color, markersize = 12)
    scatter!(ax_energy, @lift(Point2f(hours[$frame], drift[$frame])),
             color = energy_line.color, markersize = 12)
end

ax_depth.yreversed = true
axislegend(ax_depth, position = :rb)
axislegend(ax_energy, position = :rb)

save("overflow.png", fig)
@info "Saved overflow.png"

record(fig, "overflow.mp4", frames; framerate = 12) do n
    frame[] = n
end

@info "Saved overflow.mp4"

#####
##### Vorticity
#####

zᶠ = znodes(underlying_grid, f)

# A scale that leaves the strongest few per mille of the field saturated, so that the shear layer
# along the slope is visible rather than a couple of extreme points setting the range.
all_vorticity = vcat((abs.(filter(isfinite, reduce(vcat, vec.(result.vorticity_snapshots))))
                      for result in results)...)
ζmax = quantile(all_vorticity, 0.999)

vorticity_frame = Observable(last(frames))

vorticity_figure = Figure(size=(1150, 900))

for (n, result) in enumerate(results)
    title = @lift string(result.name, " bottom, t = ", prettytime(result.times[$vorticity_frame]))

    ax = Axis(vorticity_figure[n, 1:3], title = title,
              xlabel = n == length(results) ? "x (km)" : "",
              ylabel = "z (m)")

    vorticity = @lift masked_snapshot(result.vorticity_snapshots, vorticity_masks[n], $vorticity_frame)

    heatmap!(ax, xᶠ, zᶠ, vorticity, colormap = :balance, colorrange = (-ζmax, ζmax), nan_color = :gray80)
    lines!(ax, xb ./ 1e3, zb, color = :black, linewidth = 2)
    xlims!(ax, 0, xmax)
    ylims!(ax, -Lz, 0)
    n < length(results) && hidexdecorations!(ax, grid=false)
end

Colorbar(vorticity_figure[1:3, 4], colormap = :balance, limits = (-ζmax, ζmax),
         label = "∂u/∂z - ∂w/∂x (s⁻¹)")

save("overflow_vorticity.png", vorticity_figure)
@info "Saved overflow_vorticity.png"

record(vorticity_figure, "overflow_vorticity.mp4", frames; framerate = 12) do n
    vorticity_frame[] = n
end

@info "Saved overflow_vorticity.mp4"

for result in results
    drift = (result.energies[end] - result.energies[1]) / abs(result.energies[1])
    conservation = (result.buoyancies[end] - result.buoyancies[1]) / abs(result.buoyancies[1])
    @printf("%-8s  center of mass = %6.1f m   ΔRPE/RPE(0) = %.3e   Δ(∫b dV)/∫b dV = %.2e\n",
            result.name, result.depths[end], drift, conservation)
end
