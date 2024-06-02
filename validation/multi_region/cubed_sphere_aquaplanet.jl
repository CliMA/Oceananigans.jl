using Oceananigans, Printf

using Oceananigans.Grids: node, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!, Iterate
using Oceananigans.Operators
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils
using Oceananigans.TurbulenceClosures
using Oceananigans.Operators: Δx, Δy
using Oceananigans.Units

using JLD2

## Grid setup

function exponential_z_faces(p)
    Nz, Lz, k₀ = p.Nz, p.Lz, p.k₀
    
    A = [exp(-1/k₀)      1
         exp(-(Nz+1)/k₀) 1]
    
    b = [-Lz, 0]
    
    coefficients = A \ b
    
    z_faces = coefficients[1] * exp.(-(1:Nz+1)/k₀) .+ coefficients[2]
    z_faces[Nz+1] = 0

    return z_faces
end

function geometric_z_faces(p)
    Nz, Lz, ratio = p.Nz, p.Lz, p.ratio
    
    Δz = Lz * (1 - ratio) / (1 - ratio^Nz)
    
    z_faces = zeros(Nz + 1)
    
    z_faces[1] = -Lz
    for i in 2:Nz+1
        z_faces[i] = z_faces[i-1] + Δz * ratio^(i-2)
    end
    
    z_faces[Nz+1] = 0
    
    return z_faces
end

Lz = 3000
h = 0.25 * Lz

Nx, Ny, Nz = 32, 32, 20
Nhalo = 4

ratio = 0.8

φs = (-90, -45, -15, 0, 15, 45, 90)
τs = (0, 0.2, -0.1, -0.02, -0.1, 0.1, 0)

my_parameters = (Lz        = Lz,
                 h         = h,
                 Nz        = Nz,
                 k₀        = 0.25 * Nz, # Exponential profile parameter
                 ratio     = ratio,     # Geometric profile parameter
                 ρ₀        = 1020,      # Boussinesq density
                 φs        = φs,  
                 τs        = τs,
                 Δ         = 0.06,
                 φ_max_lin = 90,
                 φ_max_par = 90,
                 φ_max_cos = 90,
                 λ_rts     = 10days,    # Restoring time scale
                 Cᴰ        = 1e-3       # Drag coefficient
)

arch = CPU()
grid = ConformalCubedSphereGrid(arch;
                                panel_size = (Nx, Ny, Nz),
                                z = geometric_z_faces(my_parameters),
                                horizontal_direction_halo = Nhalo,
                                partition = CubedSpherePartition(; R = 1))

Hx, Hy, Hz = halo_size(grid)

Δz = minimum_zspacing(grid)

my_parameters = merge(my_parameters, (Δz = minimum_zspacing(grid), 𝓋 = Δz/my_parameters.λ_rts,))

@inline function cubic_interpolate(x, x₁, x₂, y₁, y₂, d₁ = 0, d₂ = 0)
    A = [x₁^3   x₁^2 x₁  1
         x₂^3   x₂^2 x₂  1
         3*x₁^2 2*x₁ 1   0
         3*x₂^2 2*x₂ 1   0]
          
    b = [y₁, y₂, d₁, d₂]

    coefficients = A \ b

    return coefficients[1] * x^3 + coefficients[2] * x^2 + coefficients[3] * x + coefficients[4]
end

# Specify the wind stress as a function of latitude, φ.
@inline function wind_stress(λ, φ, t, p) 
    φ_index = sum(φ .> p.φs) + 1
    
    φ₁ = p.φs[φ_index-1]
    φ₂ = p.φs[φ_index]
    τ₁ = p.τs[φ_index-1]
    τ₂ = p.τs[φ_index]
    
    return cubic_interpolate(φ, φ₁, φ₂, τ₁, τ₂) / p.ρ₀
end

@inline linear_profile_in_z(z, p) = 1 + z/p.Lz
@inline exponential_profile_in_z(z, p) = (exp(z/p.h) - exp(-p.Lz/p.h))/(1 - exp(-p.Lz/p.h))

@inline linear_profile_in_y(φ, p) = 1 - abs(φ)/p.φ_max_lin
@inline parabolic_profile_in_y(φ, p) = 1 - (φ/p.φ_max_par)^2
@inline cosine_profile_in_y(φ, p) = 0.5(1 + cos(π * min(max(φ/p.φ_max_cos, -1), 1)))

@inline function buoyancy_restoring(λ, φ, z, b, p)
    B = p.Δ * cosine_profile_in_y(φ, p) * linear_profile_in_z(z, p)
    return p.𝓋 * (b - B)
end

####
#### Boundary conditions
####

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

@inline speedᶠᶜᶜ(i, j, k, grid, u, v) = @inbounds sqrt(u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², v))
@inline speedᶜᶠᶜ(i, j, k, grid, u, v) = @inbounds sqrt(ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², u) + v[i, j, k]^2)

@inline u_drag(i, j, grid, clock, fields, p) = (
@inbounds - p.Cᴰ * speedᶠᶜᶜ(i, j, 1, grid, fields.u, fields.v) * fields.u[i, j, 1])
@inline v_drag(i, j, grid, clock, fields, p) = (
@inbounds - p.Cᴰ * speedᶜᶠᶜ(i, j, 1, grid, fields.u, fields.v) * fields.v[i, j, 1])

no_slip = ValueBoundaryCondition(0)
u_bcs = FieldBoundaryConditions(bottom = no_slip)
v_bcs = FieldBoundaryConditions(bottom = no_slip)
#=
u_bot_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = (; Cᴰ = my_parameters.Cᴰ))
v_bot_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = (; Cᴰ = my_parameters.Cᴰ))
top_stress_bc = FluxBoundaryCondition(wind_stress; parameters = (; φs = my_parameters.φs, τs = my_parameters.τs,
                                                                   ρ₀ = my_parameters.ρ₀)) 
u_bcs = FieldBoundaryConditions(bottom = u_bot_bc, top = top_stress_bc)
v_bcs = FieldBoundaryConditions(bottom = v_bot_bc, top = top_stress_bc)
=#

my_buoyancy_parameters = (; Δ = my_parameters.Δ, h = my_parameters.h, Lz = my_parameters.Lz,
                            φ_max_lin = my_parameters.φ_max_lin, φ_max_par = my_parameters.φ_max_par,
                            φ_max_cos = my_parameters.φ_max_cos, 𝓋 = my_parameters.𝓋)
top_restoring_bc = FluxBoundaryCondition(buoyancy_restoring; field_dependencies = :b,
                                         parameters = my_buoyancy_parameters)
b_bcs = FieldBoundaryConditions(top = top_restoring_bc)

####
#### Model setup
####

momentum_advection = VectorInvariant()
tracer_advection   = CenteredSecondOrder()
substeps           = 20
free_surface       = SplitExplicitFreeSurface(grid; substeps, extended_halos = false)

νh = 5e+4
νz = 2e-4
κh = 1e+3
κz = 2e-5

# Filter width squared, expressed as a harmonic mean of x and y spacings
@inline Δ²ᶜᶜᶜ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2
                                                      + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))

# Use a biharmonic diffusivity for momentum. Define the diffusivity function as gridsize^4 divided by the timescale.
@inline νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᶜᶜᶜ(i, j, k, grid, lx, ly, lz)^2 / λ

horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh, κ=κh)
vertical_diffusivity   = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=νz, κ=κz)
convective_adjustment  = ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), convective_κz = 1.0)
biharmonic_viscosity   = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true, parameters = (; my_parameters.λ_rts))

coriolis = HydrostaticSphericalCoriolis()

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection,
                                      tracer_advection,
                                      free_surface,
                                      coriolis,
                                      closure = (horizontal_diffusivity, vertical_diffusivity, convective_adjustment),
                                      tracers = :b,
                                      buoyancy = BuoyancyTracer(),
                                      boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs))

#####
##### Model initialization
#####

@inline initial_buoyancy(λ, φ, z) = (my_buoyancy_parameters.Δ * cosine_profile_in_y(φ, my_buoyancy_parameters)
                                     * linear_profile_in_z(z, my_buoyancy_parameters))
# Specify the initial buoyancy profile to match the buoyancy restoring profile.
set!(model, b = initial_buoyancy) 

fill_halo_regions!(model.tracers.b)

Ω = model.coriolis.rotation_rate
for region in number_of_regions(grid), k in 1:Nz, j in 1:Ny, i in 1:Nx
    numerator = model.tracers.b[region][i, j, k] - model.tracers.b[region][i, j-1, k]
    denominator = -2Ω * sind(grid[region].φᶠᶜᵃ[i, j]) * grid[region].Δyᶠᶜᵃ[i, j]
    if k == 1
        Δz = grid[region].zᵃᵃᶜ[k] - grid[region].zᵃᵃᶠ[k]
        u_below = 0 # no slip boundary condition
    else
        Δz = grid[region].Δzᵃᵃᶠ[k]
        u_below = model.velocities.u[region][i, j, k-1]
    end
    model.velocities.u[region][i, j, k] = u_below + numerator/denominator * Δz
    numerator = model.tracers.b[region][i, j, k] - model.tracers.b[region][i-1, j, k]
    denominator = 2Ω * sind(grid[region].φᶜᶠᵃ[i, j]) * grid[region].Δxᶜᶠᵃ[i, j]
    if k == 1
        v_below = 0 # no slip boundary condition
    else
        v_below = model.velocities.v[region][i, j, k-1]
    end
    model.velocities.v[region][i, j, k] = v_below + numerator/denominator * Δz
end

fill_halo_regions!((model.velocities.u, model.velocities.v))

Δt = 5minutes

stop_time = 100days
Ntime = round(Int, stop_time/Δt)

print_output_to_jld2_file = true
if print_output_to_jld2_file
    Ntime = 1500
    stop_time = Ntime * Δt
end

@info "Stop time = $(prettytime(stop_time))"
@info "Number of time steps = $Ntime"

simulation = Simulation(model; Δt, stop_time)

# Print a progress message
progress_message_iteration_interval = 10
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f, max|b|: %.3f, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u),
                                maximum(abs, model.free_surface.η) - Lz, maximum(abs, model.tracers.b),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(progress_message_iteration_interval))

u_fields = Field[]
save_u(sim) = push!(u_fields, deepcopy(sim.model.velocities.u))

v_fields = Field[]
save_v(sim) = push!(v_fields, deepcopy(sim.model.velocities.v))

# Now, compute the vorticity.
ζ = Field{Face, Face, Center}(grid)

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

offset = -1 .* halo_size(grid)

@apply_regionally begin
    kernel_parameters = KernelParameters(total_size(ζ[1]), offset)
    launch!(arch, grid, kernel_parameters, _compute_vorticity!, ζ, grid, model.velocities.u, model.velocities.v)
end

ζ_fields = Field[]

function save_ζ(sim)
    grid = sim.model.grid
    
    offset = -1 .* halo_size(grid)
    
    u, v, _ = sim.model.velocities

    fill_halo_regions!((u, v))

    @apply_regionally begin
        kernel_parameters = KernelParameters(total_size(ζ[1]), offset)
        launch!(arch, grid, kernel_parameters, _compute_vorticity!, ζ, grid, u, v)
    end

    push!(ζ_fields, deepcopy(ζ))
end

η_fields = Field[]
save_η(sim) = push!(η_fields, deepcopy(sim.model.free_surface.η))

b_fields = Field[]
save_b(sim) = push!(b_fields, deepcopy(sim.model.tracers.b))

animation_time = 15 # seconds
framerate = 5
n_frames = animation_time * framerate # excluding the initial condition frame
simulation_time_per_frame = stop_time / n_frames
save_fields_iteration_interval = floor(Int, simulation_time_per_frame/Δt)
# Redefine the simulation time per frame.
simulation_time_per_frame = save_fields_iteration_interval * Δt
# Redefine the number of frames.
n_frames = floor(Int, Ntime / save_fields_iteration_interval) # excluding the initial condition frame
# Redefine the animation time.
animation_time = n_frames / framerate
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_ζ] = Callback(save_ζ, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_η] = Callback(save_η, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_b] = Callback(save_b, IterationInterval(save_fields_iteration_interval))

run!(simulation)

if print_output_to_jld2_file
    jldopen("cubed_sphere_aquaplanet_initial_condition.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[1][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[1][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[1][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[1][region][:, :, Nz+1]
            file["b/"*string(region)] = b_fields[1][region][:, :, Nz]
        end
    end
    jldopen("cubed_sphere_aquaplanet_output.jld2", "w") do file
        for region in 1:6
            file["u/"*string(region)] = u_fields[end][region][:, :, Nz]
            file["v/"*string(region)] = v_fields[end][region][:, :, Nz]
            file["ζ/"*string(region)] = ζ_fields[end][region][:, :, Nz]
            file["η/"*string(region)] = η_fields[end][region][:, :, Nz+1]
            file["b/"*string(region)] = b_fields[end][region][:, :, Nz]
        end
    end
end

include("cubed_sphere_visualization.jl")

cos_θ, sin_θ = calculate_sines_and_cosines_of_cubed_sphere_grid_angles(grid, "cc")

function orient_velocities_in_global_direction!(grid, cos_θ, sin_θ, u_fields, v_fields; levels = 1:1)
    n_frames = length(u_fields) - 1
    for i_frame in 1:n_frames+1
        u_fields[i_frame] = interpolate_cubed_sphere_field_to_cell_centers(grid, u_fields[i_frame], "fc";
                                                                           levels = levels)
        v_fields[i_frame] = interpolate_cubed_sphere_field_to_cell_centers(grid, v_fields[i_frame], "cf";
                                                                           levels = levels)
        orient_in_global_direction!(grid, u_fields[i_frame], v_fields[i_frame], cos_θ, sin_θ; levels = levels)
    end
end

orient_panel_wise_velocity_plots_in_global_direction = true

orientation_complete = false
if orient_panel_wise_velocity_plots_in_global_direction
    orient_velocities_in_global_direction!(grid, cos_θ, sin_θ, u_fields, v_fields; levels = Nz:Nz)
    orientation_complete = true
end

for i_frame in 1:n_frames+1
    ζ_fields[i_frame] = interpolate_cubed_sphere_field_to_cell_centers(grid, ζ_fields[i_frame], "ff"; levels = Nz:Nz)
end

plot_final_field = false
if plot_final_field
    fig = panel_wise_visualization_with_halos(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_u_with_halos.png", fig)

    fig = panel_wise_visualization(grid, u_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_u.png", fig)

    fig = panel_wise_visualization_with_halos(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_v_with_halos.png", fig)

    fig = panel_wise_visualization(grid, v_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_v.png", fig)

    fig = panel_wise_visualization_with_halos(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_ζ_with_halos.png", fig)

    fig = panel_wise_visualization(grid, ζ_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_ζ.png", fig)

    fig = panel_wise_visualization_with_halos(grid, η_fields[end]; k = Nz + 1, ssh = true)
    save("cubed_sphere_aquaplanet_η_with_halos.png", fig)

    fig = panel_wise_visualization(grid, η_fields[end]; k = Nz + 1, ssh = true)
    save("cubed_sphere_aquaplanet_η.png", fig)

    fig = panel_wise_visualization_with_halos(grid, b_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_b_with_halos.png", fig)

    fig = panel_wise_visualization(grid, b_fields[end]; k = Nz)
    save("cubed_sphere_aquaplanet_b.png", fig)
end

if !orientation_complete
    orient_velocities_in_global_direction!(grid, cos_θ, sin_θ, u_fields, v_fields; levels = Nz:Nz)
end

plot_snapshots = false
if plot_snapshots
    n_snapshots = 3

    u_colorrange = zeros(2)
    v_colorrange = zeros(2)
    ζ_colorrange = zeros(2)
    η_colorrange = zeros(2)
    b_colorrange = zeros(2)

    common_kwargs = (consider_all_levels = false, vertical_dimensions = Nz:Nz)

    for i_snapshot in 0:n_snapshots
        frame_index = floor(Int, i_snapshot * n_frames / n_snapshots) + 1
        u_colorrange_at_frame_index = specify_colorrange(grid, u_fields[frame_index]; common_kwargs...)
        v_colorrange_at_frame_index = specify_colorrange(grid, v_fields[frame_index]; common_kwargs...)
        ζ_colorrange_at_frame_index = specify_colorrange(grid, ζ_fields[frame_index]; common_kwargs...)
        η_colorrange_at_frame_index = specify_colorrange(grid, η_fields[frame_index]; ssh = true)
        b_colorrange_at_frame_index = specify_colorrange(grid, b_fields[frame_index]; common_kwargs...)
        if i_snapshot == 0
            u_colorrange[:] = collect(u_colorrange_at_frame_index)
            v_colorrange[:] = collect(v_colorrange_at_frame_index)
            ζ_colorrange[:] = collect(ζ_colorrange_at_frame_index)
            η_colorrange[:] = collect(η_colorrange_at_frame_index)
            b_colorrange[:] = collect(b_colorrange_at_frame_index)
        else
            u_colorrange[1] = min(u_colorrange[1], u_colorrange_at_frame_index[1])
            u_colorrange[2] = max(u_colorrange[2], u_colorrange_at_frame_index[2])
            v_colorrange[1] = min(v_colorrange[1], v_colorrange_at_frame_index[1])
            v_colorrange[2] = max(v_colorrange[2], v_colorrange_at_frame_index[2])
            ζ_colorrange[1] = min(ζ_colorrange[1], ζ_colorrange_at_frame_index[1])
            ζ_colorrange[2] = max(ζ_colorrange[2], ζ_colorrange_at_frame_index[2])
            η_colorrange[1] = min(η_colorrange[1], η_colorrange_at_frame_index[1])
            η_colorrange[2] = max(η_colorrange[2], η_colorrange_at_frame_index[2])
            b_colorrange[1] = min(b_colorrange[1], b_colorrange_at_frame_index[1])
            b_colorrange[2] = max(b_colorrange[2], b_colorrange_at_frame_index[2])
        end
    end

    for i_snapshot in 0:n_snapshots
        frame_index = floor(Int, i_snapshot * n_frames / n_snapshots) + 1
        simulation_time = simulation_time_per_frame * (frame_index - 1)
        if i_snapshot > 0
            title = "Zonal velocity after $(prettytime(simulation_time))"
            fig = geo_heatlatlon_visualization(grid, u_fields[frame_index], title; k = Nz,
                                               cbar_label = "zonal velocity", specify_plot_limits = true,
                                               plot_limits = u_colorrange)
            save(@sprintf("cubed_sphere_aquaplanet_u_%d.png", i_snapshot), fig)
            title = "Meridional velocity after $(prettytime(simulation_time))"
            fig = geo_heatlatlon_visualization(grid, v_fields[frame_index], title; k = Nz,
                                               cbar_label = "meridional velocity", specify_plot_limits = true,
                                               plot_limits = v_colorrange)
            save(@sprintf("cubed_sphere_aquaplanet_v_%d.png", i_snapshot), fig)
            title = "Relative vorticity after $(prettytime(simulation_time))"
            fig = geo_heatlatlon_visualization(grid, ζ_fields[frame_index], title; k = Nz,
                                               cbar_label = "relative vorticity", specify_plot_limits = true,
                                               plot_limits = ζ_colorrange)
            save(@sprintf("cubed_sphere_aquaplanet_ζ_%d.png", i_snapshot), fig)
        end
        title = "Surface elevation after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, η_fields[frame_index], title; ssh = true,
                                           cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_η_%d.png", i_snapshot), fig)
        title = "Tracer distribution after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, b_fields[frame_index], title; k = Nz, cbar_label = "tracer level",
                                           specify_plot_limits = true, plot_limits = b_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_b_%d.png", i_snapshot), fig)
    end
end

make_animations = false
if make_animations
    common_kwargs = (consider_all_levels = false, vertical_dimensions = Nz:Nz)
    create_panel_wise_visualization_animation(grid, u_fields, framerate, "cubed_sphere_aquaplanet_u"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid, v_fields, framerate, "cubed_sphere_aquaplanet_v"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid, ζ_fields, framerate, "cubed_sphere_aquaplanet_ζ"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid, η_fields, framerate, "cubed_sphere_aquaplanet_η"; k = Nz+1,
                                              ssh = true)
    create_panel_wise_visualization_animation(grid, b_fields, framerate, "cubed_sphere_aquaplanet_b"; k = Nz,
                                              common_kwargs...)

    prettytimes = [prettytime(simulation_time_per_frame * i) for i in 0:n_frames]

    u_colorrange = specify_colorrange_timeseries(grid, u_fields; common_kwargs...)
    geo_heatlatlon_visualization_animation(grid, u_fields, "cc", prettytimes, "Zonal velocity",
                                           "cubed_sphere_aquaplanet_u_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "zonal velocity", specify_plot_limits = true,
                                           plot_limits = u_colorrange, framerate = framerate)

    v_colorrange = specify_colorrange_timeseries(grid, v_fields; common_kwargs...)
    geo_heatlatlon_visualization_animation(grid, v_fields, "cc", prettytimes, "Meridional velocity",
                                           "cubed_sphere_aquaplanet_v_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "meridional velocity", specify_plot_limits = true,
                                           plot_limits = v_colorrange, framerate = framerate)

    ζ_colorrange = specify_colorrange_timeseries(grid, ζ_fields; common_kwargs...)
    geo_heatlatlon_visualization_animation(grid, ζ_fields, "cc", prettytimes, "Relative vorticity",
                                           "cubed_sphere_aquaplanet_ζ_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "relative vorticity", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange, framerate = framerate)

    η_colorrange = specify_colorrange_timeseries(grid, η_fields; ssh = true)
    geo_heatlatlon_visualization_animation(grid, η_fields, "cc", prettytimes, "Surface elevation",
                                           "cubed_sphere_aquaplanet_η_geo_heatlatlon_animation"; ssh = true,
                                           cbar_label = "surface elevation", specify_plot_limits = true,
                                           plot_limits = η_colorrange, framerate = framerate)

    b_colorrange = specify_colorrange_timeseries(grid, b_fields; common_kwargs...)
    geo_heatlatlon_visualization_animation(grid, b_fields, "cc", prettytimes, "Buoyancy",
                                           "cubed_sphere_aquaplanet_b_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "buoyancy", specify_plot_limits = true,
                                           plot_limits = b_colorrange, framerate = framerate)

    Nc = Nx
    panel_index = 1 # Choose panel index to be 1 or 2.
    north_panel_index = grid.connectivity.connections[panel_index].north.from_rank
    south_panel_index = grid.connectivity.connections[panel_index].south.from_rank
    b_fields_at_specific_longitude_through_panel_center = zeros(n_frames+1, 2*Nc, Nz)
    latitudes = zeros(2*Nc)
    depths = grid[1].zᵃᵃᶜ[1:Nz]

    if panel_index == 1
        latitudes[1:round(Int, Nc/2)] = grid[south_panel_index].φᶜᶜᵃ[round(Int, Nc/2), round(Int, Nc/2)+1:Nc]
        latitudes[round(Int, Nc/2)+1:round(Int, 3Nc/2)] = grid[panel_index].φᶜᶜᵃ[round(Int, Nc/2), 1:Nc]
        latitudes[round(Int, 3Nc/2)+1:2*Nc] = grid[north_panel_index].φᶜᶜᵃ[1:round(Int, Nc/2), round(Int, Nc/2)]
        for i_frame in 1:n_frames+1
            b_fields_at_specific_longitude_through_panel_center[i_frame, 1:round(Int, Nc/2), :] = (
            b_fields[i_frame][south_panel_index][round(Int, Nc/2), round(Int, Nc/2)+1:Nc, 1:Nz])
            b_fields_at_specific_longitude_through_panel_center[i_frame, round(Int, Nc/2)+1:round(Int, 3Nc/2), :] = (
            b_fields[i_frame][panel_index][round(Int, Nc/2), 1:Nc, 1:Nz])
            b_fields_at_specific_longitude_through_panel_center[i_frame, round(Int, 3Nc/2)+1:2*Nc, :] = (
            b_fields[i_frame][north_panel_index][1:round(Int, Nc/2), round(Int, Nc/2), 1:Nz])
        end
    elseif panel_index == 2
        latitudes[1:round(Int, Nc/2)] = grid[south_panel_index].φᶜᶜᵃ[round(Int, Nc/2)+1:Nc, round(Int, Nc/2)]
        latitudes[round(Int, Nc/2)+1:round(Int, 3Nc/2)] = grid[panel_index].φᶜᶜᵃ[round(Int, Nc/2), 1:Nc]
        latitudes[round(Int, 3Nc/2)+1:2*Nc] = grid[north_panel_index].φᶜᶜᵃ[round(Int, Nc/2), 1:round(Int, Nc/2)]
        for i_frame in 1:n_frames+1
            b_fields_at_specific_longitude_through_panel_center[i_frame, 1:round(Int, Nc/2), :] = (
            b_fields[i_frame][south_panel_index][round(Int, Nc/2)+1:Nc, round(Int, Nc/2), 1:Nz])
            b_fields_at_specific_longitude_through_panel_center[i_frame, round(Int, Nc/2)+1:round(Int, 3Nc/2), :] = (
            b_fields[i_frame][panel_index][round(Int, Nc/2), 1:Nc, 1:Nz])
            b_fields_at_specific_longitude_through_panel_center[i_frame, round(Int, 3Nc/2)+1:2*Nc, :] = (
            b_fields[i_frame][north_panel_index][round(Int, Nc/2), 1:round(Int, Nc/2), 1:Nz])
        end
    end

    resolution = (875, 750)
    plot_type = "heat_map"
    axis_kwargs = (xlabel = "Latitude (degrees)", ylabel = "Depth", xlabelsize = 22.5, ylabelsize = 22.5,
                   xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1,
                   title = "Buoyancy", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    contourlevels = 50
    cbar_kwargs = (label = "buoyancy", labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
    create_heat_map_or_contour_plot(resolution, plot_type, latitudes, depths,
                                    b_fields_at_specific_longitude_through_panel_center[1, :, :], axis_kwargs,
                                    contourlevels, cbar_kwargs, "cubed_sphere_aquaplanet_b₀_latitude-depth_section")
    create_heat_map_or_contour_plot(resolution, plot_type, latitudes, depths,
                                    b_fields_at_specific_longitude_through_panel_center[end, :, :], axis_kwargs,
                                    contourlevels, cbar_kwargs, "cubed_sphere_aquaplanet_b_latitude-depth_section")
    create_heat_map_or_contour_plot_animation(resolution, plot_type, latitudes, depths,
                                              b_fields_at_specific_longitude_through_panel_center, axis_kwargs,
                                              contourlevels, cbar_kwargs, framerate,
                                              "cubed_sphere_aquaplanet_b_latitude-depth_animation";
                                              use_prettytimes = true, prettytimes = prettytimes)
end
