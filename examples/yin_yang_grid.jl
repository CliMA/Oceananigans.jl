# # Barotropic jet and Double Drake on a Yin-Yang grid
#
# This example runs two simulations on a grid that covers the whole sphere, poles included.
# The first is the unstable midlatitude jet of [Galewsky2004](@citet), a standard test of
# shallow-water models on the sphere, in a single layer of fluid. The second is the
# "Double Drake" ocean of [FerreiraMarshallCampin2010](@citet), in which two thin continents
# and a circumpolar channel shape wind-driven, baroclinically unstable fronts in both hemispheres.
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
# The Yin panel ends at 45°N, so the jet of the first simulation runs along the curve where
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
using Oceananigans.Grids: λnode, φnode, znode
using Oceananigans.Operators: rotation_angle, ζ₃ᶠᶠᶜ, ℑxyᶜᶜᵃ
using Oceananigans.MultiRegion: owned_weights
using CUDA
using Printf
using CairoMakie

# We run on the GPU. To run on the CPU, replace `GPU()` with `CPU()`: the jet then takes
# about 5 minutes, and Double Drake about 17 minutes, on a laptop with 12 threads.

arch = GPU()

# ## Barotropic jet
#
# A zonal jet at 45°N starts in balance with the height of a single layer of fluid. A small
# bump in the height disturbs the jet, which meanders and rolls up into a chain of vortices
# within six days.
#
# ### A single layer of fluid
#
# A `HydrostaticFreeSurfaceModel` with one layer, no buoyancy and a
# [`MutableVerticalDiscretization`](@ref) solves the nonlinear shallow-water equations:
# the ``z^\star`` vertical coordinate stretches the layer with the free surface ``η``, so
# the fluid depth is ``H + η``. The layer is ``H = 10`` km deep at rest.
#
# Each panel of the Earth-sized [`YinYangGrid`](@ref) has 128 cells in latitude and 384 in
# longitude, a spacing of 0.7°, or 78 km.

H = 10kilometers

jet_grid = YinYangGrid(arch; size=(384, 128, 1), z=MutableVerticalDiscretization((-H, 0)))

jet_model = HydrostaticFreeSurfaceModel(jet_grid; buoyancy=nothing, tracers=(),
                                        coriolis = HydrostaticSphericalCoriolis(),
                                        closure = HorizontalScalarDiffusivity(ν=1e5),
                                        free_surface = ExplicitFreeSurface(),
                                        momentum_advection = VectorInvariant(),
                                        timestepper = :SplitRungeKutta3)

# Following [Galewsky2004](@citet), a horizontal viscosity of ``10^5 \, \mathrm{m^2 \, s^{-1}}``
# removes the enstrophy that cascades to the grid scale.
#
# ### The jet
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
# h′ = \hat h \cos φ \, e^{-(λ / α)^2} e^{-[(φ_2 - φ) / β]^2}
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

set!(jet_model, u=uᵢ, η=ηᵢ)

# ### The simulation
#
# The free surface is explicit, so the time step must resolve gravity waves, which travel
# at ``\sqrt{g H} ≈ 313 \, \mathrm{m \, s^{-1}}``. With a time step of 40 seconds, the
# fastest waves cross about a third of the smallest cells in one step.

jet_simulation = Simulation(jet_model; Δt=40, stop_time=6days)

# The panels disagree slightly on the fluxes between them, so the volume of the fluid
# drifts. A conservation fixer removes this drift.

add_conservation_fixer!(jet_simulation)

# A progress message reports the wall time since the previous message.

wall_clock = Ref(time_ns())

function wall_time()
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    wall_clock[] = time_ns()
    return prettytime(elapsed)
end

function jet_progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("iteration: %d, time: %s, max|u|: %.1f m/s, wall time: %s",
                   iteration(sim), prettytime(sim), maximum(abs, u), wall_time())
    return nothing
end

add_callback!(jet_simulation, jet_progress, IterationInterval(1000))

# Every three hours, we save the vertical vorticity of each panel.

@inline ζᶜᶜᶜ(i, j, k, grid, u, v) = ℑxyᶜᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v)

function vertical_vorticity(model)
    u, v, w = model.velocities
    return Field(KernelFunctionOperation{Center, Center, Center}(ζᶜᶜᶜ, model.grid, u, v))
end

jet_ζ = vertical_vorticity(jet_model)

jet_times = Float64[]
vorticity = (yin = Matrix{Float64}[], yang = Matrix{Float64}[])

function save_vorticity(sim)
    compute!(jet_ζ)
    push!(jet_times, time(sim))
    push!(vorticity.yin, Array(interior(jet_ζ[1], :, :, 1)))
    push!(vorticity.yang, Array(interior(jet_ζ[2], :, :, 1)))
    return nothing
end

add_callback!(jet_simulation, save_vorticity, TimeInterval(3hours))

# We also track the volume of the fluid. The ``z^\star`` coordinate stretches the cells with
# the free surface, so the volume is the integral of one over the grid.

fluid = CenterField(jet_grid)
set!(fluid, 1)
volume = Field(Integral(fluid))
total(integral) = first(on_architecture(CPU(), interior(compute!(integral))))
initial_volume = total(volume)

run!(jet_simulation)

@info @sprintf("The relative change in volume after %s is %.1e", prettytime(jet_simulation),
               total(volume) / initial_volume - 1)

# ### Visualization
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

# CairoMakie orders the triangles of each plot by depth, but not the plots themselves.
# We therefore draw both panels as one mesh, with two triangles per quad, and give the
# four corners of each quad the value of its cell. Cells that `land` marks on a panel
# get `NaN`, which Double Drake uses below to leave its continents gray.

function panel_mesh(grid; land=nothing)
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
    masked = isnothing(land) ? falses(length(yin_cells) + length(yang_cells)) : cell_values(land(yin), land(yang))
    on_quads(yin, yang) = repeat(ifelse.(masked, NaN, cell_values(yin, yang)), inner=4)

    return (; vertices, triangles, on_quads)
end

jet_mesh = panel_mesh(jet_grid)
nothing #hide

# The box of an `Axis3` with `viewmode=:fit` keeps its size as the camera moves. We fit
# the box inside the sphere, so that the sphere fills the axis.

function globe!(position, azimuth, elevation; width=400)
    ax = Axis3(position; width, height=width, aspect=:data, azimuth, elevation,
               limits=(-0.6, 0.6, -0.6, 0.6, -0.6, 0.6), viewmode=:fit, clip=false, protrusions=0)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

jet_frame = Observable(1)
ζₙ = @lift 1e5 * jet_mesh.on_quads(vorticity.yin[$jet_frame], vorticity.yang[$jet_frame])
azimuth = @lift 2π * ($jet_frame - 1) / length(jet_times)
title = @lift "Vertical vorticity, t = " * prettytime(jet_times[$jet_frame])

colormap = :balance
colorrange = (-15, 15)

jet_fig = Figure()
Label(jet_fig[0, 1:2], title, fontsize=20, tellwidth=false)

for (column, offset) in enumerate((0, π))
    ax = globe!(jet_fig[1, column], @lift($azimuth + offset), π/4; width=450)
    mesh!(ax, jet_mesh.vertices, jet_mesh.triangles; color=ζₙ, colormap, colorrange, shading=NoShading)
end

Colorbar(jet_fig[2, 1:2]; colormap, colorrange, label="Vertical vorticity (10⁻⁵ s⁻¹)",
         vertical=false, width=500)

resize_to_layout!(jet_fig)

CairoMakie.record(jet_fig, "yin_yang_barotropic_jet.mp4", eachindex(jet_times), framerate=8) do i
    jet_frame[] = i
end
nothing #hide

# ![](yin_yang_barotropic_jet.mp4)
#
# By day 4 the jet meanders, and by day 6 it has rolled up into a chain of vortices.
#
# ## Double Drake
#
# The second simulation has wind-driven, baroclinically unstable fronts in an ocean that
# covers the whole sphere. Two thin continents run from the North Pole to 35°S along 0°
# and 90°E. They divide the ocean into a small and a large basin, which connect through a
# circumpolar channel in the south. This is the "Double Drake" geometry of
# [FerreiraMarshallCampin2010](@citet), an idealized Atlantic and Pacific with a Drake Passage.
#
# ### The grid
#
# To resolve baroclinic eddies with a modest number of cells, we use a small planet with
# a radius of 1000 km that rotates at Earth's rate. Each panel has 64 cells in latitude
# and 192 in longitude, a spacing of 1.4°, or 25 km. Four layers span 4 km of ocean.
# We build the panels with [`YinYangGrid`](@ref), and use a
# [`MutableVerticalDiscretization`](@ref) for a ``z^\star`` vertical coordinate.

radius = 1000kilometers
z_faces = [-4000, -1500, -500, -100, 0]
Nz = length(z_faces) - 1

underlying_grid = YinYangGrid(arch; size=(192, 64, Nz), halo=(5, 5, 4), radius,
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

drake_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

# ### Wind stress
#
# Westerlies blow over the mid-latitudes and trade winds over the tropics, with the
# strongest winds over the Southern Ocean:
#
# ```math
# τ(φ) = τ₀ \, \cos φ \left [2 e^{-(φ + 50^\circ)^2 / (12^\circ)^2} + e^{-(φ - 45^\circ)^2 / (12^\circ)^2}
#        - 0.7 e^{-(φ + 18^\circ)^2 / (12^\circ)^2} - 0.6 e^{-(φ - 18^\circ)^2 / (12^\circ)^2} \right ] ,
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

# ### Fronts in both hemispheres
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
# ``P(λ) = \cos(6λ) + \cos(9λ) / 2 + \sin(13λ) / 4``, with amplitude ``ϵ = 0.035`` and
# ``m = \sin^2 φ \cos^2 φ / (\sin^2 φ_0 \cos^2 φ_0)``, seed the instability.
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

# ### The model
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

drake_model = HydrostaticFreeSurfaceModel(drake_grid; buoyancy,
                                          coriolis = HydrostaticSphericalCoriolis(),
                                          closure = (vertical_mixing, horizontal_mixing),
                                          free_surface = SplitExplicitFreeSurface(substeps=12, extend_halos=false),
                                          momentum_advection = WENOVectorInvariant(order=5),
                                          tracer_advection = WENO(order=5, bounds=(front.Td, front.Td + front.ΔT)),
                                          tracers = :T,
                                          timestepper = :SplitRungeKutta3,
                                          vertical_coordinate = ZStarCoordinate(),
                                          boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs))

# ### A balanced initial state
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

set!(drake_model, u = (λ, φ, z) -> initial_state(λ, φ, z).u,
                  v = (λ, φ, z) -> initial_state(λ, φ, z).v,
                  T = (λ, φ, z) -> initial_state(λ, φ, z).T,
                  η = (λ, φ, z) -> initial_state(λ, φ, z).η)

# ### The simulation
#
# We run for twenty days with a time step of 150 seconds.

drake_simulation = Simulation(drake_model; Δt=150, stop_time=20days)

# The panels disagree slightly on the fluxes between them, so a conservation fixer removes the volume and heat they add.

add_conservation_fixer!(drake_simulation)

# The progress message also reports the vertical velocity.

function drake_progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("iteration: %d, time: %s, max|u|: %.2f m/s, max|w|: %.1e m/s, wall time: %s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w), wall_time())
    return nothing
end

add_callback!(drake_simulation, drake_progress, IterationInterval(1000))

# For plotting, we save the surface vorticity and temperature of each panel every three hours.

drake_ζ = vertical_vorticity(drake_model)

drake_times = Float64[]
surface_ζ = (yin = Matrix{Float64}[], yang = Matrix{Float64}[])
surface_T = (yin = Matrix{Float64}[], yang = Matrix{Float64}[])

function save_surface_fields(sim)
    compute!(drake_ζ)
    T = sim.model.tracers.T
    push!(drake_times, time(sim))
    push!(surface_ζ.yin, Array(interior(drake_ζ[1], :, :, Nz)))
    push!(surface_ζ.yang, Array(interior(drake_ζ[2], :, :, Nz)))
    push!(surface_T.yin, Array(interior(T[1], :, :, Nz)))
    push!(surface_T.yang, Array(interior(T[2], :, :, Nz)))
    return nothing
end

add_callback!(drake_simulation, save_surface_fields, TimeInterval(3hours))

# We also track the heat content, ``∫ T \, \mathrm{d}V``, which changes only through surface
# restoring, since the conservation fixer removes the small errors of the interpolation between the panels.

heat_content = Field(Integral(drake_model.tracers.T))
initial_heat_content = total(heat_content)

# We restart the wall clock and run.

wall_clock[] = time_ns()

run!(drake_simulation)

@info @sprintf("The heat content changed by %.3f%% in %s", 100 * (total(heat_content) / initial_heat_content - 1),
               prettytime(drake_simulation))

# ### Visualization
#
# We plot the surface fields on the sphere, seen from the north and from the south, while
# the camera circles the globe once, with the same mesh of quads as for the jet. Land is gray.

drake_mesh = panel_mesh(drake_grid; land = panel -> interior(bottom_height_field(panel), :, :, 1) .≥ 0)

drake_frame = Observable(1)
ζₙ = @lift 1e5 * drake_mesh.on_quads(surface_ζ.yin[$drake_frame], surface_ζ.yang[$drake_frame])
Tₙ = @lift drake_mesh.on_quads(surface_T.yin[$drake_frame], surface_T.yang[$drake_frame])
azimuth = @lift π/4 + 2π * ($drake_frame - 1) / length(drake_times)
title = @lift "Surface vorticity and temperature, t = " * prettytime(drake_times[$drake_frame])

drake_fig = Figure()
Label(drake_fig[0, 1:2], title, fontsize=20, tellwidth=false)

views = ((1, "From the north", +π/6),
         (2, "From the south", -π/6))

for (column, name, elevation) in views
    Label(drake_fig[1, column], name, fontsize=16, tellwidth=false)
end

rows = ((ζₙ, :balance, (-10, 10), "Vertical vorticity (10⁻⁵ s⁻¹)"),
        (Tₙ, :thermal, (0, 24), "Temperature (ᵒC)"))

for (row, (values, colormap, colorrange, label)) in enumerate(rows)
    for (column, name, elevation) in views
        ax = globe!(drake_fig[row + 1, column], azimuth, elevation)
        mesh!(ax, drake_mesh.vertices, drake_mesh.triangles; color=values, colormap, colorrange, nan_color=:gray70, shading=NoShading)
    end
    Colorbar(drake_fig[row + 1, 3]; colormap, colorrange, label)
end

resize_to_layout!(drake_fig)

CairoMakie.record(drake_fig, "yin_yang_double_drake.mp4", eachindex(drake_times), framerate=12) do i
    drake_frame[] = i
end
nothing #hide

# ![](yin_yang_double_drake.mp4)
