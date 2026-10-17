# # Double Drake on a Yin-Yang grid
#
# This example simulates wind-driven, baroclinically unstable fronts in an ocean that
# covers the whole sphere. Two thin continents run from the North Pole to 35°S along 0°
# and 90°E. They divide the ocean into a small and a large basin, which connect through a
# circumpolar channel in the south. This is the "Double Drake" geometry of
# [FerreiraMarshallCampin2010](@citet), an idealized Atlantic and Pacific with a Drake Passage.
#
# We use the Yin-Yang grid of [KageyamaSato2004](@citet), which covers the poles with two
# overlapping latitude-longitude panels, as described in the
# [barotropic jet example](yin_yang_barotropic_jet.md).
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
using Oceananigans.Grids: λnode, φnode, znode
using Oceananigans.Operators: rotation_angle, ζ₃ᶠᶠᶜ, ℑxyᶜᶜᵃ
using Oceananigans.MultiRegion: owned_weights
using CUDA
using Printf
using CairoMakie

# We run on the GPU. To run on the CPU, replace `GPU()` with `CPU()`: the simulation then
# takes about 17 minutes on a laptop with 12 threads.

arch = GPU()

# ## The grid
#
# To resolve baroclinic eddies with a modest number of cells, we use a small planet with
# a radius of 1000 km that rotates at Earth's rate. Each panel has 64 cells in latitude
# and 192 in longitude, a spacing of 1.4°, or 25 km. Four layers span 4 km of ocean.
# We build the panels with [`YinYangGrid`](@ref), and use a
# [`MutableVerticalDiscretization`](@ref) for a ``z^\star`` vertical coordinate.

Nφ = 64
radius = 1000kilometers
z_faces = [-4000, -1500, -500, -100, 0]
Nz = length(z_faces) - 1

underlying_grid = YinYangGrid(arch; size=(3Nφ, Nφ, Nz), halo=(5, 5, 4), radius,
                              z=MutableVerticalDiscretization(z_faces))

# The continents are 16° wide strips along 0° and 90°E. They join a polar cap north of
# 82°N and end at 35°S. Their width is measured from the plane of each meridian, so they
# do not thin toward the pole.

function continent(λ, φ)
    x, y = cosd(φ) * cosd(λ), cosd(φ) * sind(λ)
    along_0 = (abs(y) < sind(8)) & (x > 0)
    along_90E = (abs(x) < sind(8)) & (y > 0)
    return (φ > 82) | ((along_0 | along_90E) & (φ > -35))
end

bottom_height(λ, φ) = ifelse(continent(λ, φ), 0, -4000)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

# ## Wind stress
#
# Westerlies blow over the mid-latitudes and trade winds over the tropics, with the
# strongest winds over the Southern Ocean:
#
# ```math
# τ(φ) = τ₀ \cos φ \left [2 e^{-(φ + 50°)^2 / (12°)^2} + e^{-(φ - 45°)^2 / (12°)^2}
#        - 0.7 e^{-(φ + 18°)^2 / (12°)^2} - 0.6 e^{-(φ - 18°)^2 / (12°)^2} \right ] ,
# ```
#
# with ``τ₀ = 0.15 \, \mathrm{N \, m^{-2}}``.

@inline function zonal_wind_stress(φ)
    westerlies = 2exp(-((φ + 50) / 12)^2) + exp(-((φ - 45) / 12)^2)
    trades = 0.7exp(-((φ + 18) / 12)^2) + 0.6exp(-((φ - 18) / 12)^2)
    return 0.15 * cos(deg2rad(φ)) * (westerlies - trades)
end

# Each panel stores velocities in its own basis, which is rotated by an angle ``θ`` from
# the geographic basis. We rotate the eastward stress into the panel basis. The momentum
# flux is minus the stress divided by the reference density.

@inline function u_wind_flux(i, j, grid, clock, fields, ρ₀)
    φ = φnode(i, j, grid.Nz, grid, Face(), Center(), Center())
    θ = rotation_angle(i, j, grid, Face(), Center())
    return - zonal_wind_stress(φ) * cos(θ) / ρ₀
end

@inline function v_wind_flux(i, j, grid, clock, fields, ρ₀)
    φ = φnode(i, j, grid.Nz, grid, Center(), Face(), Center())
    θ = rotation_angle(i, j, grid, Center(), Face())
    return - zonal_wind_stress(φ) * sin(θ) / ρ₀
end

ρ₀ = 1020 # kg m⁻³
u_wind = FluxBoundaryCondition(u_wind_flux; discrete_form=true, parameters=ρ₀)
v_wind = FluxBoundaryCondition(v_wind_flux; discrete_form=true, parameters=ρ₀)

# A quadratic [`BulkDrag`](@ref) acts on the sea floor.

drag = BulkDrag(coefficient=2.5e-3)
u_bcs = FieldBoundaryConditions(top=u_wind, bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))
v_bcs = FieldBoundaryConditions(top=v_wind, bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))

# ## Fronts in both hemispheres
#
# A warm thermocline fills the top kilometer between two fronts at ``φ₀ = ±57°``, over a
# cold deep ocean and cold polar water:
#
# ```math
# T = T_d + \frac{ΔT}{2} e^{z / h} (1 - \tanh χ) , \qquad
# χ = \frac{\sin^2 φ - \sin^2 φ₀ - ϵ \, m(φ) P(λ)}{δ} ,
# ```
#
# with ``T_d = 2°``C, ``ΔT = 20°``C, ``h = 1`` km and ``δ = 0.1``. The meanders
# ``P(λ) = \cos 6λ + \cos(9λ) / 2 + \sin(13λ) / 4``, with amplitude ``ϵ = 0.035`` and
# ``m = \sin^2 φ \cos^2 φ / (\sin^2 φ₀ \cos^2 φ₀)``, seed the instability.
#
# The surface layer is restored to the unperturbed fronts, with ``ϵ = 0``, on a time scale
# of 30 days.

front = (; Td=2, ΔT=20, h=1000, φ₀=57, δ=0.1)

@inline function temperature_flux(i, j, grid, clock, fields, p)
    φ = φnode(i, j, grid.Nz, grid, Center(), Center(), Center())
    z = znode(i, j, grid.Nz, grid, Center(), Center(), Center())
    χ = (sin(deg2rad(φ))^2 - p.sin²φ₀) / p.δ
    T★ = p.Td + p.ΔT / 2 * exp(z / p.h) * (1 - tanh(χ))
    return @inbounds p.rate * (fields.T[i, j, grid.Nz] - T★)
end

restoring_rate = 100 / 30days # surface layer thickness over the restoring time [m s⁻¹]
restoring_parameters = (; front.Td, front.ΔT, front.h, front.δ, sin²φ₀=sind(front.φ₀)^2, rate=restoring_rate)
T_restoring = FluxBoundaryCondition(temperature_flux; discrete_form=true, parameters=restoring_parameters)
T_bcs = FieldBoundaryConditions(top=T_restoring)

# ## The model
#
# Convective adjustment mixes temperature and momentum where the stratification is
# unstable, and a weak background diffusivity and viscosity act everywhere. Both are
# integrated implicitly in the vertical.

vertical_mixing = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1, convective_νz=1,
                                                          background_κz=1e-5, background_νz=1e-3)

horizontal_mixing = HorizontalScalarDiffusivity(ν=100, κ=20)

α = 2e-4 # thermal expansion coefficient [K⁻¹]
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=α), constant_salinity=35)

# Where the fronts meet the continents, they drive coastal currents of about 6 m s⁻¹. With
# WENO alone, the sharp fronts these currents carry would undershoot ``T_d``. The initial state
# and the restoring lie between ``T_d`` and ``T_d + ΔT``, so we bound the advection to that range.
#
# On a Yin-Yang grid, the split-explicit free surface substeps without extending its halos.

model = HydrostaticFreeSurfaceModel(grid; buoyancy,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    closure = (vertical_mixing, horizontal_mixing),
                                    free_surface = SplitExplicitFreeSurface(substeps=12, extend_halos=false),
                                    momentum_advection = WENOVectorInvariant(order=5),
                                    tracer_advection = WENO(order=5, bounds=(front.Td, front.Td + front.ΔT)),
                                    tracers = :T,
                                    timestepper = :SplitRungeKutta3,
                                    vertical_coordinate = ZStarCoordinate(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs))

# ## A balanced initial state
#
# The fronts start in balance with an initial current. The deepest layer is at rest:
# the free surface ``η`` cancels the bottom pressure gradient of the warm water above it,
# so that the pressure anomaly at each depth is
#
# ```math
# p = - \frac{g α ΔT}{2} \tanh χ \, I(z) , \qquad I(z) = ∫_{z_1}^{z} e^{z' / h} \, \mathrm{d} z' ,
# ```
#
# where ``z_1`` is the center of the deepest cell. We integrate ``I`` with the same
# trapezoidal rule between cell centers as the model's hydrostatic pressure. The zonal
# velocity is in gradient-wind balance with ``p``, and the meridional velocity, which
# only the meanders induce, is geostrophic.

g = Oceananigans.defaults.gravitational_acceleration
Ω = Oceananigans.defaults.planet_rotation_rate
A = g * α * front.ΔT / 2

zᶜ = (z_faces[1:end-1] .+ z_faces[2:end]) ./ 2
eᶜ = exp.(zᶜ ./ front.h)
I(z) = sum((eᶜ[k] + eᶜ[k+1]) / 2 * clamp(z - zᶜ[k], 0, zᶜ[k+1] - zᶜ[k]) for k in 1:Nz-1)

# The pressure at the top cell center includes buoyancy integrated over the distance from
# that center to its mirror image above the surface:

Iˢ = I(0) - 2zᶜ[end] * eᶜ[end]

function initial_state(λ, φ, z; ϵ=0.035, φ₀=front.φ₀, δ=front.δ)
    s, c = sind(φ), cosd(φ)
    D = sind(φ₀)^2 * cosd(φ₀)^2
    λ = deg2rad(λ)
    P = cos(6λ) + cos(9λ) / 2 + sin(13λ) / 4
    ∂λP = - 6sin(6λ) - 9sin(9λ) / 2 + 13cos(13λ) / 4
    χ = (s^2 - sind(φ₀)^2 - ϵ * s^2 * c^2 * P / D) / δ

    a = A * I(z) * sech(χ)^2 / (Ω * radius * δ)
    U = a * (1 - ϵ * (c^2 - s^2) * P / D)
    u = 2c * U / (1 + sqrt(1 + 2U / (Ω * radius)))
    v = a * ϵ * s * c * ∂λP / (2D)
    T = front.Td + front.ΔT / 2 * exp(z / front.h) * (1 - tanh(χ))
    η = - A * Iˢ * tanh(χ) / g

    return (; u, v, T, η)
end

# `set!` takes the geographic components of the velocity and rotates them into each panel's basis.

set!(model, u = (λ, φ, z) -> initial_state(λ, φ, z).u,
            v = (λ, φ, z) -> initial_state(λ, φ, z).v,
            T = (λ, φ, z) -> initial_state(λ, φ, z).T,
            η = (λ, φ, z) -> initial_state(λ, φ, z).η)

# ## The simulation
#
# We run for twenty days with a time step of 150 seconds.

simulation = Simulation(model; Δt=150, stop_time=20days)

# The panels disagree slightly on the fluxes between them, so a conservation fixer removes the volume and heat they add.

add_conservation_fixer!(simulation)

wall_clock = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.velocities
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("iteration: %d, time: %s, max|u|: %.2f m/s, max|w|: %.1e m/s, wall time: %s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w), prettytime(elapsed))
    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# For plotting, we save the surface vorticity and temperature of each panel every three hours.

@inline ζᶜᶜᶜ(i, j, k, grid, u, v) = ℑxyᶜᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v)

u, v, w = model.velocities
ζ = Field(KernelFunctionOperation{Center, Center, Center}(ζᶜᶜᶜ, grid, u, v))

times = Float64[]
surface_ζ = (yin = Matrix{Float64}[], yang = Matrix{Float64}[])
surface_T = (yin = Matrix{Float64}[], yang = Matrix{Float64}[])

function save_surface_fields(sim)
    compute!(ζ)
    T = sim.model.tracers.T
    push!(times, time(sim))
    push!(surface_ζ.yin, Array(interior(ζ[1], :, :, Nz)))
    push!(surface_ζ.yang, Array(interior(ζ[2], :, :, Nz)))
    push!(surface_T.yin, Array(interior(T[1], :, :, Nz)))
    push!(surface_T.yang, Array(interior(T[2], :, :, Nz)))
    return nothing
end

add_callback!(simulation, save_surface_fields, TimeInterval(3hours))

# We also track the heat content, ``∫ T \, \mathrm{d}V``, which changes only through surface
# restoring, since the conservation fixer removes the small errors of the interpolation between the panels.

heat_content = Field(Integral(model.tracers.T))
total(integral) = first(on_architecture(CPU(), interior(compute!(integral))))
initial_heat_content = total(heat_content)

run!(simulation)

@info @sprintf("The heat content changed by %.3f%% in %s", 100 * (total(heat_content) / initial_heat_content - 1),
               prettytime(simulation))

# ## Visualization
#
# We plot the surface fields on the sphere, seen from the north and from the south, while
# the camera circles the globe once. As in the [barotropic jet example](yin_yang_barotropic_jet.md),
# each cell is drawn as a quad of one color between the corners of its panel: Yin draws the
# cells it owns and Yang, just beneath, the cells in which it owns any area, in one mesh so
# that CairoMakie orders their triangles by depth. Land is gray.

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

vertices = vcat(yin_vertices, yang_vertices)
quads = 1:length(vertices) ÷ 4
triangles = [[4q-3 for q in quads] [4q-2 for q in quads] [4q-1 for q in quads];
             [4q-3 for q in quads] [4q-1 for q in quads] [4q   for q in quads]]

cell_values(yin, yang) = vcat(yin[yin_cells], yang[yang_cells])
land = cell_values(interior(bottom_height_field(yin), :, :, 1), interior(bottom_height_field(yang), :, :, 1)) .≥ 0
on_quads(yin, yang) = repeat(ifelse.(land, NaN, cell_values(yin, yang)), inner=4)

function globe!(position, azimuth, elevation)
    ax = Axis3(position; width=400, height=400, aspect=:data, azimuth, elevation,
               limits=(-0.6, 0.6, -0.6, 0.6, -0.6, 0.6), viewmode=:fit, clip=false, protrusions=0)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

n = Observable(1)
ζₙ = @lift 1e5 * on_quads(surface_ζ.yin[$n], surface_ζ.yang[$n])
Tₙ = @lift on_quads(surface_T.yin[$n], surface_T.yang[$n])
azimuth = @lift π/4 + 2π * ($n - 1) / length(times)
title = @lift "Surface vorticity and temperature, t = " * prettytime(times[$n])

fig = Figure()
Label(fig[0, 1:2], title, fontsize=20, tellwidth=false)

views = ((1, "From the north", +π/6),
         (2, "From the south", -π/6))

for (column, name, elevation) in views
    Label(fig[1, column], name, fontsize=16, tellwidth=false)
end

rows = ((ζₙ, :balance, (-10, 10), "Vertical vorticity (10⁻⁵ s⁻¹)"),
        (Tₙ, :thermal, (0, 24), "Temperature (ᵒC)"))

for (row, (values, colormap, colorrange, label)) in enumerate(rows)
    for (column, name, elevation) in views
        ax = globe!(fig[row + 1, column], azimuth, elevation)
        mesh!(ax, vertices, triangles; color=values, colormap, colorrange, nan_color=:gray70, shading=NoShading)
    end
    Colorbar(fig[row + 1, 3]; colormap, colorrange, label)
end

resize_to_layout!(fig)

CairoMakie.record(fig, "yin_yang_double_drake.mp4", eachindex(times), framerate=12) do i
    n[] = i
end
nothing #hide

# ![](yin_yang_double_drake.mp4)
