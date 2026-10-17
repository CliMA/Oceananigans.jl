# # Barotropic jet on a Yin-Yang grid
#
# This example simulates the unstable midlatitude jet of [Galewsky2004](@citet), a standard
# test of shallow-water models on the sphere. A zonal jet at 45°N starts in balance with the
# height of a single layer of fluid. A small bump in the height disturbs the jet, which
# meanders and rolls up into a chain of vortices within six days.
#
# ## The Yin-Yang grid
#
# A latitude-longitude grid cannot include the poles, where its meridians converge. The
# Yin-Yang grid of [KageyamaSato2004](@citet) avoids the poles with two identical
# latitude-longitude panels, Yin and Yang. Each panel spans 270° of longitude and 90° of
# latitude around its own equator, where the grid is nearly uniform. Yang is Yin rotated so
# that its equator passes through the geographic poles. The two panels overlap at their
# edges, and each one fills its halos by interpolating from the interior of the other.
#
# A curve through the overlap divides the sphere between the panels, and integrals and
# averages count every point of the sphere once, on the side of the curve that owns it.
# The Yin panel ends at 45°N, so the jet in this example runs along the curve where
# the panels meet.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie, CUDA"
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Operators: ζ₃ᶠᶠᶜ, ℑxyᶜᶜᵃ
using Oceananigans.MultiRegion: owned_weights
using CUDA
using Printf
using CairoMakie

# We run on the GPU. To run on the CPU, replace `GPU()` with `CPU()`: the simulation then
# takes about 5 minutes on a laptop with 12 threads.

arch = GPU()

# ## A single layer of fluid
#
# A `HydrostaticFreeSurfaceModel` with one layer, no buoyancy and a
# [`MutableVerticalDiscretization`](@ref) solves the nonlinear shallow-water equations:
# the ``z^\star`` vertical coordinate stretches the layer with the free surface ``η``, so
# the fluid depth is ``H + η``. The layer is ``H = 10`` km deep at rest.
#
# Each panel of the Earth-sized [`YinYangGrid`](@ref) has 128 cells in latitude and 384 in
# longitude, a spacing of 0.7°, or 78 km.

Nφ = 128
H = 10kilometers

grid = YinYangGrid(arch; size=(3Nφ, Nφ, 1), z=MutableVerticalDiscretization((-H, 0)))

model = HydrostaticFreeSurfaceModel(grid; buoyancy=nothing, tracers=(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    closure = HorizontalScalarDiffusivity(ν=1e5),
                                    free_surface = ExplicitFreeSurface(),
                                    momentum_advection = VectorInvariant(),
                                    timestepper = :SplitRungeKutta3)

# Following [Galewsky2004](@citet), a horizontal viscosity of ``10^5 \, \mathrm{m^2 \, s^{-1}}``
# removes the enstrophy that cascades to the grid scale.
#
# ## The jet
#
# The zonal velocity is
#
# ```math
# u(φ) = \frac{u_{\max}}{e_n} \exp \left [ \frac{1}{(φ - φ₀)(φ - φ₁)} \right ]
# \quad \text{for} \quad φ₀ < φ < φ₁ ,
# ```
#
# and zero elsewhere, with ``u_{\max} = 80 \, \mathrm{m \, s^{-1}}``, ``φ₀ = π/7``,
# ``φ₁ = π/2 - φ₀`` and ``e_n = \exp[-4 / (φ₁ - φ₀)^2]``, so that the jet peaks at 45°N.

u_max = 80
φ₀ = π / 7
φ₁ = π / 2 - φ₀
e_n = exp(-4 / (φ₁ - φ₀)^2)

jet(φ) = φ₀ < φ < φ₁ ? u_max / e_n * exp(1 / ((φ - φ₀) * (φ - φ₁))) : 0.0

# The height ``h`` of the fluid is in gradient-wind balance with the jet,
#
# ```math
# g \frac{\mathrm{d} h}{\mathrm{d} φ} = - R u \left ( f + \frac{u \tan φ}{R} \right ) ,
# ```
#
# where ``R`` is the radius of the Earth and ``f = 2 Ω \sin φ``. We integrate this balance
# across the jet with the midpoint rule, and remove the global mean of the result from the
# free surface, so that the mean depth is ``H``.

g = Oceananigans.defaults.gravitational_acceleration
Ω = Oceananigans.defaults.planet_rotation_rate
R = Oceananigans.defaults.planet_radius

function integral(f, a, b; n=500)
    Δ = (b - a) / n
    return Δ * sum(f(a + (k - 1/2) * Δ) for k in 1:n)
end

dhdφ(φ) = - R * jet(φ) * (2Ω * sin(φ) + jet(φ) * tan(φ) / R) / g
height_departure(φ) = integral(dhdφ, φ₀, clamp(φ, φ₀, φ₁))
mean_departure = integral(φ -> height_departure(φ) * cos(φ), -π/2, π/2) / 2

# To destabilize the jet, we add the bump
#
# ```math
# h′ = \hat h \cos φ \, e^{-(λ / α)^2} e^{-[(φ₂ - φ) / β]^2}
# ```
#
# with ``\hat h = 120`` m, ``φ₂ = π/4``, ``α = 1/3`` and ``β = 1/15``, for ``-π < λ < π``.

function bump(λ, φ; ĥ=120, φ₂=π/4, α=1/3, β=1/15)
    λ = mod(λ + π, 2π) - π
    return ĥ * cos(φ) * exp(-(λ / α)^2) * exp(-((φ₂ - φ) / β)^2)
end

# The functions above take radians, and `set!` passes longitude and latitude in degrees.
# `set!` also rotates the eastward velocity into the basis of each panel.

uᵢ(λ, φ, z) = jet(deg2rad(φ))
ηᵢ(λ, φ, z) = height_departure(deg2rad(φ)) - mean_departure + bump(deg2rad(λ), deg2rad(φ))

set!(model, u=uᵢ, η=ηᵢ)

# ## The simulation
#
# The free surface is explicit, so the time step must resolve gravity waves, which travel
# at ``\sqrt{g H} ≈ 313 \, \mathrm{m \, s^{-1}}``. With a time step of 40 seconds, the
# fastest waves cross about a third of the smallest cells in one step.

simulation = Simulation(model; Δt=40, stop_time=6days)

# The panels disagree slightly on the fluxes between them, so the volume of the fluid
# drifts. A conservation fixer removes this drift.

add_conservation_fixer!(simulation)

wall_clock = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.velocities
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("iteration: %d, time: %s, max|u|: %.1f m/s, wall time: %s",
                   iteration(sim), prettytime(sim), maximum(abs, u), prettytime(elapsed))
    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# Every three hours, we save the vertical vorticity of each panel.

@inline ζᶜᶜᶜ(i, j, k, grid, u, v) = ℑxyᶜᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v)

u, v, w = model.velocities
ζ = Field(KernelFunctionOperation{Center, Center, Center}(ζᶜᶜᶜ, grid, u, v))

times = Float64[]
vorticity = (yin = Matrix{Float64}[], yang = Matrix{Float64}[])

function save_vorticity(sim)
    compute!(ζ)
    push!(times, time(sim))
    push!(vorticity.yin, Array(interior(ζ[1], :, :, 1)))
    push!(vorticity.yang, Array(interior(ζ[2], :, :, 1)))
    return nothing
end

add_callback!(simulation, save_vorticity, TimeInterval(3hours))

# We also track the volume of the fluid. The ``z^\star`` coordinate stretches the cells with
# the free surface, so the volume is the integral of one over the grid.

fluid = CenterField(grid)
set!(fluid, 1)
volume = Field(Integral(fluid))
total(integral) = first(on_architecture(CPU(), interior(compute!(integral))))
initial_volume = total(volume)

run!(simulation)

@info @sprintf("The relative change in volume after %s is %.1e", prettytime(simulation),
               total(volume) / initial_volume - 1)

# ## Visualization
#
# We plot the vorticity on the sphere, seen from two sides, while the camera circles the
# globe once. Each cell is drawn as a quad of one color between the corners of its panel,
# so the plot shows the solution as the panels hold it. Yin draws the cells it owns, and
# Yang, just beneath, the cells in which it owns any area, so that the overlap appears once.

sphere_point(λ, φ, r) = Point3f(r * cosd(φ) * cosd(λ), r * cosd(φ) * sind(λ), r * sind(φ))

function cell_quads(panel, drawn; radius=1)
    Nz = size(panel, 3)
    corner(i, j) = sphere_point(λnode(i, j, Nz, panel, Face(), Face(), Center()),
                                φnode(i, j, Nz, panel, Face(), Face(), Center()), radius)
    cells = findall(drawn)
    vertices = [corner(c[1] + di, c[2] + dj) for c in cells for (di, dj) in ((0, 0), (1, 0), (1, 1), (0, 1))]
    return vertices, cells
end

yin = on_architecture(CPU(), grid[1])
yang = on_architecture(CPU(), grid[2])
weights = owned_weights(grid)
yin_vertices, yin_cells = cell_quads(yin, Array(interior(weights[1], :, :, 1)) .≥ 1/2)
yang_vertices, yang_cells = cell_quads(yang, Array(interior(weights[2], :, :, 1)) .> 0; radius=0.995)

# CairoMakie orders the triangles of each plot by depth, but not the plots themselves.
# We therefore draw both panels as one mesh, with two triangles per quad, and give the
# four corners of each quad the value of its cell.

vertices = vcat(yin_vertices, yang_vertices)
quads = 1:length(vertices) ÷ 4
triangles = [[4q-3 for q in quads] [4q-2 for q in quads] [4q-1 for q in quads];
             [4q-3 for q in quads] [4q-1 for q in quads] [4q   for q in quads]]

on_quads(yin, yang) = repeat(vcat(yin[yin_cells], yang[yang_cells]), inner=4)

# The box of an `Axis3` with `viewmode=:fit` keeps its size as the camera moves. We fit
# the box inside the sphere, so that the sphere fills the axis.

function globe!(position, azimuth, elevation)
    ax = Axis3(position; width=450, height=450, aspect=:data, azimuth, elevation,
               limits=(-0.6, 0.6, -0.6, 0.6, -0.6, 0.6), viewmode=:fit, clip=false, protrusions=0)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

n = Observable(1)
ζₙ = @lift 1e5 * on_quads(vorticity.yin[$n], vorticity.yang[$n])
azimuth = @lift 2π * ($n - 1) / length(times)
title = @lift "Vertical vorticity, t = " * prettytime(times[$n])

colormap = :balance
colorrange = (-15, 15)

fig = Figure()
Label(fig[0, 1:2], title, fontsize=20, tellwidth=false)

for (column, offset) in enumerate((0, π))
    ax = globe!(fig[1, column], @lift($azimuth + offset), π/4)
    mesh!(ax, vertices, triangles; color=ζₙ, colormap, colorrange, shading=NoShading)
end

Colorbar(fig[2, 1:2]; colormap, colorrange, label="Vertical vorticity (10⁻⁵ s⁻¹)",
         vertical=false, width=500)

resize_to_layout!(fig)

CairoMakie.record(fig, "yin_yang_barotropic_jet.mp4", eachindex(times), framerate=8) do i
    n[] = i
end
nothing #hide

# ![](yin_yang_barotropic_jet.mp4)
#
# By day 4 the jet meanders, and by day 6 it has rolled up into a chain of vortices.
