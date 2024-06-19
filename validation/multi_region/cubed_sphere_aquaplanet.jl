using Oceananigans, Printf

using Oceananigans.Grids: node, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!, Iterate
using Oceananigans.Operators
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils
using Oceananigans.TurbulenceClosures
using Oceananigans.Operators: Δx, Δy
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries

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

φ_max_τ = 70
φs = (-φ_max_τ, -45, -15, 0, 15, 45, φ_max_τ)
τs = (0, 0.2, -0.1, -0.02, -0.1, 0.2, 0)

my_parameters = (Lz          = Lz,
                 h           = h,
                 Nz          = Nz,
                 k₀          = 0.25 * Nz, # Exponential profile parameter
                 ratio       = ratio,     # Geometric profile parameter
                 ρ₀          = 1020,      # Boussinesq density
                 φ_max_τ     = φ_max_τ,
                 φs          = φs,
                 τs          = τs,
                 Δ           = 0.06,
                 φ_max_b_lin = 90,
                 φ_max_b_par = 90,
                 φ_max_b_cos = 75,
                 λ_rts       = 10days,    # Restoring time scale
                 Cᴰ          = 1e-3       # Drag coefficient
)

radius = 6371e3
f₀ = 1e-4
L_d = (2/f₀ * sqrt(my_parameters.h * my_parameters.Δ/(1 - exp(-my_parameters.Lz/my_parameters.h)))
       * (1 - exp(-my_parameters.Lz/(2my_parameters.h))))
print("For an initial buoyancy profile decaying exponetially with depth, the Rossby radius of deformation is $L_d m.\n")
Nx_min = ceil(Int, 2π * radius/(4L_d))
print("The minimum number of grid points in each direction of the cubed sphere panels required to resolve this " *
      "Rossby radius of deformation is $(Nx_min).\n")

arch = CPU()
underlying_grid = ConformalCubedSphereGrid(arch;
                                           panel_size = (Nx, Ny, Nz),
                                           z = geometric_z_faces(my_parameters),
                                           horizontal_direction_halo = Nhalo,
                                           radius,
                                           partition = CubedSpherePartition(; R = 1))

max_spacing_degree = rad2deg(maximum(underlying_grid[1].Δxᶠᶠᵃ)/radius)

@inline function double_drake_depth(λ, φ)
    if (-40 < φ ≤ 90) && ((-max_spacing_degree < λ ≤ 0) || (90 ≤ λ < (90 + max_spacing_degree)))
        depth = 0
    else
        depth = -Lz
    end
    return depth
end

double_drake = false
grid = double_drake ? ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(double_drake_depth)) : underlying_grid;

Hx, Hy, Hz = halo_size(grid)

Δz_min = minimum_zspacing(underlying_grid)
my_parameters = merge(my_parameters, (Δz = Δz_min, 𝓋 = Δz_min/my_parameters.λ_rts,))

@inline function cubic_interpolate(x, x₁, x₂, y₁, y₂, d₁ = 0, d₂ = 0)
    A = [x₁^3   x₁^2 x₁  1
         x₂^3   x₂^2 x₂  1
         3*x₁^2 2*x₁ 1   0
         3*x₂^2 2*x₂ 1   0]
          
    b = [y₁, y₂, d₁, d₂]

    coefficients = A \ b

    return coefficients[1] * x^3 + coefficients[2] * x^2 + coefficients[3] * x + coefficients[4]
end

using Oceananigans.Grids: λnode, φnode

# Specify the wind stress as a function of latitude, φ.
@inline function wind_stress_x(i, j, grid, clock, fields, p)
    φ = φnode(i, j, 1, grid, Face(), Center(), Center())

    if abs(φ) > p.φ_max_τ
        τₓ_latlon = 0
    else
        φ_index = sum(φ .> p.φs) + 1

        φ₁ = p.φs[φ_index-1]
        φ₂ = p.φs[φ_index]
        τ₁ = p.τs[φ_index-1]
        τ₂ = p.τs[φ_index]

        τₓ_latlon = -cubic_interpolate(φ, φ₁, φ₂, τ₁, τ₂) / p.ρ₀
    end

    # Now, calculate the cosine of the angle with respect to the geographic north, and use it to determine the component
    # of τₓ_latlon in the local x direction of the cubed sphere panel.

    φᶠᶠᵃ_i_jp1 = φnode(i, j+1, 1, grid,   Face(),   Face(), Center())
    φᶠᶠᵃ_i_j   = φnode(i,   j, 1, grid,   Face(),   Face(), Center())
    Δyᶠᶜᵃ_i_j  =    Δy(i,   j, 1, grid,   Face(), Center(), Center())

    u_Pseudo = deg2rad(φᶠᶠᵃ_i_jp1 - φᶠᶠᵃ_i_j)/Δyᶠᶜᵃ_i_j

    φᶜᶜᵃ_i_j   = φnode(i,   j, 1, grid, Center(), Center(), Center())
    φᶜᶜᵃ_im1_j = φnode(i-1, j, 1, grid, Center(), Center(), Center())
    Δxᶠᶜᵃ_i_j  =    Δx(i,   j, 1, grid,   Face(), Center(), Center())

    v_Pseudo = -deg2rad(φᶜᶜᵃ_i_j - φᶜᶜᵃ_im1_j)/Δxᶠᶜᵃ_i_j

    cos_θ = u_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)

    τₓ_x = τₓ_latlon * cos_θ

    return τₓ_x
end

@inline function wind_stress_y(i, j, grid, clock, fields, p)
    φ = φnode(i, j, 1, grid, Center(), Face(), Center())
    
    if abs(φ) > p.φ_max_τ
        τₓ_latlon = 0
    else
        φ_index = sum(φ .> p.φs) + 1

        φ₁ = p.φs[φ_index-1]
        φ₂ = p.φs[φ_index]
        τ₁ = p.τs[φ_index-1]
        τ₂ = p.τs[φ_index]

        τₓ_latlon = -cubic_interpolate(φ, φ₁, φ₂, τ₁, τ₂) / p.ρ₀
    end

    # Now, calculate the sine of the angle with respect to the geographic north, and use it to determine the component
    # of τₓ_latlon in the local y direction of the cubed sphere panel.

    φᶜᶜᵃ_i_j   = φnode(i,   j, 1, grid, Center(), Center(), Center())
    φᶜᶜᵃ_i_jm1 = φnode(i, j-1, 1, grid, Center(), Center(), Center())
    Δyᶜᶠᵃ_i_j  =    Δy(i,   j, 1, grid, Center(),   Face(), Center())

    u_Pseudo = deg2rad(φᶜᶜᵃ_i_j - φᶜᶜᵃ_i_jm1)/Δyᶜᶠᵃ_i_j

    φᶠᶠᵃ_ip1_j = φnode(i+1, j, 1, grid,   Face(),   Face(), Center())
    φᶠᶠᵃ_i_j   = φnode(i,   j, 1, grid,   Face(),   Face(), Center())
    Δxᶜᶠᵃ_i_j  =    Δx(i,   j, 1, grid, Center(),   Face(), Center())

    v_Pseudo = -deg2rad(φᶠᶠᵃ_ip1_j - φᶠᶠᵃ_i_j)/Δxᶜᶠᵃ_i_j

    sin_θ = v_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)

    τₓ_y = τₓ_latlon * sin_θ

    return τₓ_y
end

@inline linear_profile_in_z(z, p) = 1 + z/p.Lz
@inline exponential_profile_in_z(z, Lz, h) = (exp(z / h) - exp(- Lz / h)) / (1 - exp(- Lz / h))

@inline linear_profile_in_y(φ, p) = 1 - abs(φ)/p.φ_max_b_lin
@inline parabolic_profile_in_y(φ, p) = 1 - (φ/p.φ_max_b_par)^2
@inline cosine_profile_in_y(φ, p) = 0.5(1 + cos(π * min(max(φ/p.φ_max_b_cos, -1), 1)))
@inline double_cosine_profile_in_y(φ, p) = (
0.5(1 + cos(π * min(max((deg2rad(abs(φ)) - π/4)/(deg2rad(p.φ_max_b_cos) - π/4), -1), 1))))

@inline function buoyancy_restoring(λ, φ, t, b, p)
    B = p.Δ * cosine_profile_in_y(φ, p)
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

u_bot_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = (; Cᴰ = my_parameters.Cᴰ))
v_bot_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = (; Cᴰ = my_parameters.Cᴰ))
top_stress_x = FluxBoundaryCondition(wind_stress_x; discrete_form = true,
                                     parameters = (; φ_max_τ = my_parameters.φ_max_τ, φs = my_parameters.φs,
                                                     τs = my_parameters.τs, ρ₀ = my_parameters.ρ₀))
top_stress_y = FluxBoundaryCondition(wind_stress_y; discrete_form = true,
                                     parameters = (; φ_max_τ = my_parameters.φ_max_τ, φs = my_parameters.φs,
                                                     τs = my_parameters.τs, ρ₀ = my_parameters.ρ₀))
u_bcs = FieldBoundaryConditions(bottom = u_bot_bc, top = top_stress_x)
v_bcs = FieldBoundaryConditions(bottom = v_bot_bc, top = top_stress_y)

my_buoyancy_parameters = (; Δ = my_parameters.Δ, h = my_parameters.h, Lz = my_parameters.Lz,
                            φ_max_b_lin = my_parameters.φ_max_b_lin, φ_max_b_par = my_parameters.φ_max_b_par,
                            φ_max_b_cos = my_parameters.φ_max_b_cos, 𝓋 = my_parameters.𝓋)
top_restoring_bc = FluxBoundaryCondition(buoyancy_restoring; field_dependencies = :b,
                                         parameters = my_buoyancy_parameters)
b_bcs = FieldBoundaryConditions(top = top_restoring_bc)

####
#### Model setup
####

momentum_advection = VectorInvariant()
tracer_advection   = WENO()
substeps           = 20
free_surface       = SplitExplicitFreeSurface(grid; substeps, extended_halos = false)

# Filter width squared, expressed as a harmonic mean of x and y spacings
@inline Δ²ᶜᶜᶜ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2
                                                      + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))

# Use a biharmonic viscosity for momentum. Define the viscosity function as gridsize^4 divided by the timescale.
@inline νhb(i, j, k, grid, lx, ly, lz, clock, fields, p) = Δ²ᶜᶜᶜ(i, j, k, grid, lx, ly, lz)^2 / p.λ_rts

biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = νhb, discrete_form = true,
                                                             parameters = (; λ_rts = my_parameters.λ_rts))

κh = 1e+3
horizontal_diffusivity = HorizontalScalarDiffusivity(κ = κh) # Laplacian viscosity and diffusivity

νz_surface = 5e-3
νz_bottom = 1e-4

struct MyViscosity{FT} <: Function
    Lz  :: FT
    h   :: FT
    νzs :: FT
    νzb :: FT
end

using Adapt

Adapt.adapt_structure(to, ν::MyViscosity) = MyViscosity(Adapt.adapt(to, ν.Lz),  Adapt.adapt(to, ν.h),
                                                        Adapt.adapt(to, ν.νzs), Adapt.adapt(to, ν.νzb))

@inline (ν::MyViscosity)(x, y, z, t) = ν.νzb + (ν.νzs - ν.νzb) * exponential_profile_in_z(z, ν.Lz, ν.h)

νz = MyViscosity(float(Lz), h, νz_surface, νz_bottom)

κz = 2e-5

vertical_diffusivity  = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = νz, κ = κz)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization(),
                                                                convective_κz = 1.0)

coriolis = HydrostaticSphericalCoriolis()

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection,
                                      tracer_advection,
                                      free_surface,
                                      coriolis,
                                      closure = (horizontal_diffusivity, biharmonic_viscosity, vertical_diffusivity,
                                                 convective_adjustment),
                                      tracers = :b,
                                      buoyancy = BuoyancyTracer(),
                                      boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs))

#####
##### Model initialization
#####

@inline initial_buoyancy(λ, φ, z) = (my_buoyancy_parameters.Δ * cosine_profile_in_y(φ, my_buoyancy_parameters)
                                     * exponential_profile_in_z(z, my_parameters.Lz, my_parameters.h))
# Specify the initial buoyancy profile to match the buoyancy restoring profile.
set!(model, b = initial_buoyancy) 

initialize_velocities_based_on_thermal_wind_balance = false
# If the above flag is set to true, meaning the velocities are initialized using thermal wind balance, set
# φ_max_b_cos within the range [70, 80], and specify the latitudinal variation in buoyancy as
# p.Δ * double_cosine_profile_in_y(φ, p) in both the initial buoyancy and the surface buoyancy restoring profiles.
if initialize_velocities_based_on_thermal_wind_balance
    fill_halo_regions!(model.tracers.b)

    Ω = model.coriolis.rotation_rate
    radius = grid.radius

    for region in 1:number_of_regions(grid), k in 1:Nz, j in 1:Ny, i in 1:Nx
        numerator = model.tracers.b[region][i, j, k] - model.tracers.b[region][i, j-1, k]
        denominator = -2Ω * sind(grid[region].φᶠᶜᵃ[i, j]) * grid[region].Δyᶠᶜᵃ[i, j]
        if k == 1
            Δz_below = grid[region].zᵃᵃᶜ[k] - grid[region].zᵃᵃᶠ[k]
            u_below = 0 # no slip boundary condition
        else
            Δz_below = grid[region].Δzᵃᵃᶠ[k]
            u_below = model.velocities.u[region][i, j, k-1]
        end
        model.velocities.u[region][i, j, k] = u_below + numerator/denominator * Δz_below
        numerator = model.tracers.b[region][i, j, k] - model.tracers.b[region][i-1, j, k]
        denominator = 2Ω * sind(grid[region].φᶜᶠᵃ[i, j]) * grid[region].Δxᶜᶠᵃ[i, j]
        if k == 1
            v_below = 0 # no slip boundary condition
        else
            v_below = model.velocities.v[region][i, j, k-1]
        end
        model.velocities.v[region][i, j, k] = v_below + numerator/denominator * Δz_below
    end

    fill_halo_regions!((model.velocities.u, model.velocities.v))
end

# Compute the initial vorticity.
ζ = Field{Face, Face, Center}(grid)

offset = -1 .* halo_size(grid)

fill_halo_regions!((model.velocities.u, model.velocities.v))

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

@apply_regionally begin
    kernel_parameters = KernelParameters(total_size(ζ[1]), offset)
    launch!(arch, grid, kernel_parameters, _compute_vorticity!, ζ, grid, model.velocities.u, model.velocities.v)
end

# Compute actual and reconstructed wind stress.
τₓ = CenterField(grid, indices = (1:Nx, 1:Ny, 1:1))
τ_xr = CenterField(grid, indices = (1:Nx, 1:Ny, 1:1)) # Reconstructed zonal wind stress
τ_yr = CenterField(grid, indices = (1:Nx, 1:Ny, 1:1)) # Reconstructed meridional wind stress, expected to be zero

for region in 1:number_of_regions(grid), j in 1:Ny, i in 1:Nx
    φ = φnode(i, j, 1, grid[region], Center(), Center(), Center())

    if abs(φ) > my_parameters.φ_max_τ
        τₓ[region][i, j, 1] = 0
    else
        φ_index = sum(φ .> my_parameters.φs) + 1

        φ₁ = my_parameters.φs[φ_index-1]
        φ₂ = my_parameters.φs[φ_index]
        τ₁ = my_parameters.τs[φ_index-1]
        τ₂ = my_parameters.τs[φ_index]

        τₓ[region][i, j, 1] = -cubic_interpolate(φ, φ₁, φ₂, τ₁, τ₂) / my_parameters.ρ₀
    end

    φᶜᶠᵃ_i_jp1 = φnode(i, j+1, 1, grid[region], Center(),   Face(), Center())
    φᶜᶠᵃ_i_j   = φnode(i,   j, 1, grid[region], Center(),   Face(), Center())
    Δyᶜᶜᵃ_i_j  =    Δy(i,   j, 1, grid[region], Center(), Center(), Center())

    u_Pseudo = deg2rad(φᶜᶠᵃ_i_jp1 - φᶜᶠᵃ_i_j)/Δyᶜᶜᵃ_i_j

    φᶠᶜᵃ_ip1_j = φnode(i+1, j, 1, grid[region],   Face(), Center(), Center())
    φᶠᶜᵃ_i_j   = φnode(i,   j, 1, grid[region],   Face(), Center(), Center())
    Δxᶜᶜᵃ_i_j  =    Δx(i,   j, 1, grid[region], Center(), Center(), Center())

    v_Pseudo = -deg2rad(φᶠᶜᵃ_ip1_j - φᶠᶜᵃ_i_j)/Δxᶜᶜᵃ_i_j

    cos_θ = u_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)
    sin_θ = v_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)

    τₓ_x = τₓ[region][i, j, 1] * cos_θ
    τₓ_y = τₓ[region][i, j, 1] * sin_θ

    τ_xr[region][i, j, 1] = τₓ_x * cos_θ + τₓ_y * sin_θ
    τ_yr[region][i, j, 1] = τₓ_y * cos_θ - τₓ_x * sin_θ
end

# Plot wind stress and initial fields.
uᵢ = deepcopy(model.velocities.u)
vᵢ = deepcopy(model.velocities.v)
ζᵢ = deepcopy(ζ)
bᵢ = deepcopy(model.tracers.b)

include("cubed_sphere_visualization.jl")

latitude = extract_latitude(grid)
cos_θ, sin_θ = calculate_sines_and_cosines_of_cubed_sphere_grid_angles(grid, "cc")

cos_θ_at_specific_longitude_through_panel_center    = zeros(2*Nx, 4);
sin_θ_at_specific_longitude_through_panel_center    = zeros(2*Nx, 4);
latitude_at_specific_longitude_through_panel_center = zeros(2*Nx, 4);

for (index, panel_index) in enumerate([1])
    cos_θ_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(grid, cos_θ, panel_index))
    sin_θ_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(grid, sin_θ, panel_index))
    latitude_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(grid, latitude, panel_index))
end

depths = grid[1].zᵃᵃᶜ[1:Nz]

uᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
vᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
ζᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
bᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);

resolution = (875, 750)
plot_type_1D = "scatter_line_plot"
plot_kwargs = (linewidth = 2, linecolor = :black, marker = :rect, markersize = 10)
plot_type_2D = "heat_map"
axis_kwargs = (xlabel = "Latitude (degrees)", ylabel = "Depth (km)", xlabelsize = 22.5, ylabelsize = 22.5,
               xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1,
               titlesize = 27.5, titlegap = 15, titlefont = :bold)
axis_kwargs_ssh = (; axis_kwargs..., ylabel = "Surface elevation (m)")
contourlevels = 50
cbar_kwargs = (labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
common_kwargs = (; consider_all_levels = false)
b_index = round(Int, Nz/2)
common_kwargs_geo_colorrange = (consider_all_levels = false, levels = Nz:Nz)
common_kwargs_geo_colorrange_b = (consider_all_levels = false, levels = b_index:b_index)
common_kwargs_geo_τ = (consider_all_levels = false, levels = 1:1)
common_kwargs_geo = (consider_all_levels = false, k = Nz)
common_kwargs_geo_b = (consider_all_levels = false, k = b_index)

plot_initial_field = true
if plot_initial_field
    fig = panel_wise_visualization(grid, τₓ; k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress.png", fig)

    fig = panel_wise_visualization(grid, τ_xr; k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed.png", fig)

    fig = panel_wise_visualization(grid, τ_yr; k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed.png", fig)

    title = "Zonal wind stress"
    fig = geo_heatlatlon_visualization(grid, τₓ, title; common_kwargs_geo_τ...,
                                       cbar_label = "zonal wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_zonal_wind_stress_geo_heatlatlon_plot.png", fig)

    title = "Reconstructed zonal wind stress"
    fig = geo_heatlatlon_visualization(grid, τ_xr, title; common_kwargs_geo_τ...,
                                       cbar_label = "zonal wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)

    title = "Reconstructed meridional wind stress"
    fig = geo_heatlatlon_visualization(grid, τ_yr, title; common_kwargs_geo_τ...,
                                       cbar_label = "meridional wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)

    if initialize_velocities_based_on_thermal_wind_balance
        uᵢ, vᵢ = orient_velocities_in_global_direction(grid, uᵢ, vᵢ, cos_θ, sin_θ; levels = 1:Nz)

        fig = panel_wise_visualization(grid, uᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_uᵢ.png", fig)

        fig = panel_wise_visualization(grid, vᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_vᵢ.png", fig)

        ζᵢ = interpolate_cubed_sphere_field_to_cell_centers(grid, ζᵢ, "ff"; levels = 1:Nz)

        fig = panel_wise_visualization(grid, ζᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_ζᵢ.png", fig)

        for (index, panel_index) in enumerate([1])
            uᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, uᵢ, panel_index; levels = 1:Nz))
            vᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, vᵢ, panel_index; levels = 1:Nz))
            ζᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, ζᵢ, panel_index; levels = 1:Nz))

            title = "Zonal velocity"
            cbar_label = "zonal velocity (m s⁻¹)"
            create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                            latitude_at_specific_longitude_through_panel_center[:, index],
                                            depths/1000, uᵢ_at_specific_longitude_through_panel_center[:, :, index],
                                            axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                            "cubed_sphere_aquaplanet_uᵢ_latitude-depth_section_$panel_index" )
            title = "Meridional velocity"
            cbar_label = "meridional velocity (m s⁻¹)"
            create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                            latitude_at_specific_longitude_through_panel_center[:, index],
                                            depths/1000, vᵢ_at_specific_longitude_through_panel_center[:, :, index],
                                            axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                            "cubed_sphere_aquaplanet_vᵢ_latitude-depth_section_$panel_index")
            title = "Relative vorticity"
            cbar_label = "relative vorticity (s⁻¹)"
            create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                            latitude_at_specific_longitude_through_panel_center[:, index],
                                            depths/1000, ζᵢ_at_specific_longitude_through_panel_center[:, :, index],
                                            axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                            "cubed_sphere_aquaplanet_ζᵢ_latitude-depth_section_$panel_index")
        end

        title = "Initial zonal velocity"
        fig = geo_heatlatlon_visualization(grid, uᵢ, title; common_kwargs_geo..., cbar_label = "zonal velocity (m s⁻¹)")
        save("cubed_sphere_aquaplanet_u_0.png", fig)

        title = "Initial meridional velocity"
        fig = geo_heatlatlon_visualization(grid, vᵢ, title; common_kwargs_geo...,
                                           cbar_label = "meridional velocity (m s⁻¹)")
        save("cubed_sphere_aquaplanet_v_0.png", fig)

        title = "Initial relative vorticity"
        fig = geo_heatlatlon_visualization(grid, ζᵢ, title; common_kwargs_geo...,
                                           cbar_label = "relative vorticity (s⁻¹)")
        save("cubed_sphere_aquaplanet_ζ_0.png", fig)
    end

    fig = panel_wise_visualization(grid, bᵢ; k = b_index, common_kwargs...)

    save("cubed_sphere_aquaplanet_bᵢ.png", fig)
    for (index, panel_index) in enumerate([1])
        bᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid, bᵢ, panel_index; levels = 1:Nz))
        title = "Buoyancy"
        cbar_label = "buoyancy (m s⁻²)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index], depths/1000,
                                        bᵢ_at_specific_longitude_through_panel_center[:, :, index], axis_kwargs, title,
                                        contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_bᵢ_latitude-depth_section_$panel_index")
    end

    title = "Initial buoyancy"
    fig = geo_heatlatlon_visualization(grid, bᵢ, title; common_kwargs_geo_b..., cbar_label = "buoyancy (m s⁻²)")
    save("cubed_sphere_aquaplanet_b_0.png", fig)
end

#####
##### Simulation setup
#####

Δt = 5minutes

min_spacing = filter(!iszero, grid[1].Δxᶠᶠᵃ) |> minimum
c = sqrt(model.free_surface.gravitational_acceleration * Lz)
CourantNumber = 0.25
min_substeps = ceil(Int, c * Δt / (CourantNumber * min_spacing))
print("The minimum number of substeps required to satisfy the CFL condition is $min_substeps.\n")

debug_mode = false
if debug_mode
    stop_time = 2days
    save_fields_interval = 6hours
    checkpointer_interval = 12hours
else
    month = 30days
    months = month
    year = 365days
    years = year
    stop_time = 100years
    save_fields_interval = 1month
    checkpointer_interval = 1year
end
# Note that n_frames = floor(Int, stop_time/save_fields_interval) + 1.

Ntime = round(Int, stop_time/Δt)

@info "Stop time = $(prettytime(stop_time))"
@info "Number of time steps = $Ntime"

simulation = Simulation(model; Δt, stop_time)

# Print a progress message.
progress_message_iteration_interval = 10
progress_message(sim) = (
@printf("Iteration: %04d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f, max|b|: %.3f, wall time: %s\n",
        iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u),
        maximum(abs, model.free_surface.η) - Lz, maximum(abs, model.tracers.b), prettytime(sim.run_wall_time)))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(progress_message_iteration_interval))

#####
##### Build checkpointer and output writer
#####

pick_up_simulation = false
if pick_up_simulation
    pick_up = (pickup = true)
    overwrite_existing_output_writer = (overwrite_existing = false)
else
    pick_up = (pickup = false)
    overwrite_existing_output_writer = (overwrite_existing = true)
end

filename_checkpointer = "cubed_sphere_aquaplanet_checkpointer"
simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(checkpointer_interval),
                                                        prefix = filename_checkpointer,
                                                        overwrite_existing = true)

ζ = Oceananigans.Models.HydrostaticFreeSurfaceModels.VerticalVorticityField(model)

outputs = merge(fields(model), (; ζ))
filename_output_writer = "cubed_sphere_aquaplanet_output"
simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs;
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = filename_output_writer,
                                                      verbose = false,
                                                      overwrite_existing = overwrite_existing_output_writer...)

#####
##### Run simulation
#####

@info "Running the simulation..."

run!(simulation, pick_up...)

u_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "u");
v_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "v");
ζ_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "ζ");
η_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "η");
b_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "b");

x_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "b");

n_frames = length(u_timeseries)

for i_frame in 1:n_frames
    u_frame, v_frame = (
    orient_velocities_in_global_direction(grid, u_timeseries[i_frame], v_timeseries[i_frame], cos_θ, sin_θ;
                                          levels = 1:Nz))
    ζ_frame = interpolate_cubed_sphere_field_to_cell_centers(grid, ζ_timeseries[i_frame], "ff"; levels = 1:Nz)
    set!(u_timeseries[i_frame], u_frame)
    set!(v_timeseries[i_frame], v_frame)
    set!(ζ_timeseries[i_frame], ζ_frame)
end

u_f_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
v_f_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
ζ_f_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
η_f_at_specific_longitude_through_panel_center = zeros(2*Nx,  1, 4);
b_f_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);

plot_final_field = true
if plot_final_field
    fig = panel_wise_visualization(grid, u_timeseries[end]; k = Nz, common_kwargs...)
    save("cubed_sphere_aquaplanet_u_f.png", fig)

    fig = panel_wise_visualization(grid, v_timeseries[end]; k = Nz, common_kwargs...)
    save("cubed_sphere_aquaplanet_v_f.png", fig)

    fig = panel_wise_visualization(grid, ζ_timeseries[end]; k = Nz, common_kwargs...)
    save("cubed_sphere_aquaplanet_ζ_f.png", fig)

    fig = panel_wise_visualization(grid, η_timeseries[end]; ssh = true)
    save("cubed_sphere_aquaplanet_η_f.png", fig)

    fig = panel_wise_visualization(grid, b_timeseries[end]; k = b_index, common_kwargs...)
    save("cubed_sphere_aquaplanet_b_f.png", fig)

    for (index, panel_index) in enumerate([1])
        u_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid, u_timeseries[end], panel_index; levels = 1:Nz))
        v_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid, v_timeseries[end], panel_index; levels = 1:Nz))
        ζ_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid, ζ_timeseries[end], panel_index; levels = 1:Nz))
        η_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid, η_timeseries[end], panel_index;
                                                                 levels = Nz+1:Nz+1))
        b_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid, b_timeseries[end], panel_index; levels = 1:Nz))
        title = "Zonal velocity"
        cbar_label = "zonal velocity (m s⁻¹)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index],
                                        depths/1000, u_f_at_specific_longitude_through_panel_center[:, :, index],
                                        axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_u_f_latitude-depth_section_$panel_index")
        title = "Meridional velocity"
        cbar_label = "meridional velocity (m s⁻¹)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index],
                                        depths/1000, v_f_at_specific_longitude_through_panel_center[:, :, index],
                                        axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_v_f_latitude-depth_section_$panel_index")
        title = "Relative vorticity"
        cbar_label = "relative vorticity (s⁻¹)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index],
                                        depths/1000, ζ_f_at_specific_longitude_through_panel_center[:, :, index],
                                        axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_ζ_f_latitude-depth_section_$panel_index")
        title = "Surface elevation"
        cbar_label = "surface elevation (m)"
        create_single_line_or_scatter_plot(resolution, plot_type_1D,
                                           latitude_at_specific_longitude_through_panel_center[:, index],
                                           η_f_at_specific_longitude_through_panel_center[:, 1, index], axis_kwargs_ssh,
                                           title, plot_kwargs, "cubed_sphere_aquaplanet_η_f_latitude_$panel_index";
                                           tight_x_axis = true)
        title = "Buoyancy"
        cbar_label = "buoyancy (m s⁻²)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index],
                                        depths/1000, b_f_at_specific_longitude_through_panel_center[:, :, index],
                                        axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_b_f_latitude-depth_section_$panel_index")
    end
end

plot_snapshots = true
if plot_snapshots
    n_snapshots = 4 + 1
    Δn_snapshots = floor(Int, (n_frames - 1)/(n_snapshots - 1))
    # Ensure that (n_frames - 1) is divisible by (n_snapshots - 1).

    u_colorrange = specify_colorrange_timeseries(grid, u_timeseries; common_kwargs_geo_colorrange..., Δ = Δn_snapshots)
    v_colorrange = specify_colorrange_timeseries(grid, v_timeseries; common_kwargs_geo_colorrange..., Δ = Δn_snapshots)
    ζ_colorrange = specify_colorrange_timeseries(grid, ζ_timeseries; common_kwargs_geo_colorrange..., Δ = Δn_snapshots)
    η_colorrange = specify_colorrange_timeseries(grid, η_timeseries; ssh = true, Δ = Δn_snapshots)
    b_colorrange = specify_colorrange_timeseries(grid, b_timeseries; common_kwargs_geo_colorrange_b...,
                                                 Δ = Δn_snapshots)

    for i_snapshot in 1:(n_snapshots - 1)
        frame_index = floor(Int, i_snapshot * (n_frames - 1)/(n_snapshots - 1) + 1)
        simulation_time = (frame_index - 1) * save_fields_interval

        title = "Zonal velocity after $(prettytime(simulation_time))"
        set!(x_timeseries[frame_index], u_timeseries[frame_index])
        fig = geo_heatlatlon_visualization(grid, x_timeseries[frame_index], title; common_kwargs_geo...,
                                           cbar_label = "zonal velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = u_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_u_%d.png", i_snapshot), fig)

        title = "Meridional velocity after $(prettytime(simulation_time))"
        set!(x_timeseries[frame_index], v_timeseries[frame_index])
        fig = geo_heatlatlon_visualization(grid, x_timeseries[frame_index], title; common_kwargs_geo...,
                                           cbar_label = "meridional velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = v_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_v_%d.png", i_snapshot), fig)

        title = "Relative vorticity after $(prettytime(simulation_time))"
        set!(x_timeseries[frame_index], ζ_timeseries[frame_index])
        fig = geo_heatlatlon_visualization(grid, x_timeseries[frame_index], title; common_kwargs_geo...,
                                           cbar_label = "relative vorticity (s⁻¹)", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_ζ_%d.png", i_snapshot), fig)

        title = "Surface elevation after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, η_timeseries[frame_index], title; ssh = true,
                                           cbar_label = "surface elevation (m)", specify_plot_limits = true,
                                           plot_limits = η_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_η_%d.png", i_snapshot), fig)

        title = "Buoyancy after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid, b_timeseries[frame_index], title; common_kwargs_geo_b...,
                                           cbar_label = "buoyancy (m s⁻²)", specify_plot_limits = true,
                                           plot_limits = b_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_b_%d.png", i_snapshot), fig)
    end
end

function copy_to_center_field(x_timeseries, y_timeseries)
    n_frames = length(x_timeseries)
    for i_frame in 1:n_frames
        set!(x_timeseries[i_frame], y_timeseries[i_frame])
    end
end

make_animations = true
if make_animations
    animation_time = 9 # seconds
    framerate = floor(Int, n_frames/animation_time)
    # Redefine the animation time.
    animation_time = n_frames / framerate

    create_panel_wise_visualization_animation(grid, u_timeseries, framerate, "cubed_sphere_aquaplanet_u"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid, v_timeseries, framerate, "cubed_sphere_aquaplanet_v"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid, ζ_timeseries, framerate, "cubed_sphere_aquaplanet_ζ"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid, η_timeseries, framerate, "cubed_sphere_aquaplanet_η"; ssh = true)
    create_panel_wise_visualization_animation(grid, b_timeseries, framerate, "cubed_sphere_aquaplanet_b"; k = b_index,
                                              common_kwargs...)

    prettytimes = [prettytime((i - 1) * save_fields_interval) for i in 1:n_frames]

    u_at_specific_longitude_through_panel_center = zeros(n_frames, 2*Nx, Nz, 4);
    v_at_specific_longitude_through_panel_center = zeros(n_frames, 2*Nx, Nz, 4);
    ζ_at_specific_longitude_through_panel_center = zeros(n_frames, 2*Nx, Nz, 4);
    η_at_specific_longitude_through_panel_center = zeros(n_frames, 2*Nx,  1, 4);
    b_at_specific_longitude_through_panel_center = zeros(n_frames, 2*Nx, Nz, 4);

    for (index, panel_index) in enumerate([1])
        for i_frame in 1:n_frames
            u_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, u_timeseries[i_frame], panel_index;
                                                                     levels = 1:Nz))

            v_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, v_timeseries[i_frame], panel_index;
                                                                     levels = 1:Nz))

            ζ_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, ζ_timeseries[i_frame], panel_index;
                                                                     levels = 1:Nz))

            η_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, η_timeseries[i_frame], panel_index;
                                                                     levels = Nz+1:Nz+1))

            b_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid, b_timeseries[i_frame], panel_index;
                                                                     levels = 1:Nz))
        end

        title = "Zonal velocity"
        cbar_label = "zonal velocity (m s⁻¹)"
        create_heat_map_or_contour_plot_animation(resolution, plot_type_2D,
                                                  latitude_at_specific_longitude_through_panel_center[:, index],
                                                  depths/1000, u_at_specific_longitude_through_panel_center[:, :, :,
                                                                                                            index],
                                                  axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label, framerate,
                                                  "cubed_sphere_aquaplanet_u_latitude-depth_section_$panel_index";
                                                  use_prettytimes = true, prettytimes = prettytimes)

        title = "Meridional velocity"
        cbar_label = "meridional velocity (m s⁻¹)"
        create_heat_map_or_contour_plot_animation(resolution, plot_type_2D,
                                                  latitude_at_specific_longitude_through_panel_center[:, index],
                                                  depths/1000, v_at_specific_longitude_through_panel_center[:, :, :,
                                                                                                            index],
                                                  axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label, framerate,
                                                  "cubed_sphere_aquaplanet_v_latitude-depth_section_$panel_index";
                                                  use_prettytimes = true, prettytimes = prettytimes)

        title = "Relative vorticity"
        cbar_label = "relative vorticity (s⁻¹)"
        create_heat_map_or_contour_plot_animation(resolution, plot_type_2D,
                                                  latitude_at_specific_longitude_through_panel_center[:, index],
                                                  depths/1000, ζ_at_specific_longitude_through_panel_center[:, :, :,
                                                                                                            index],
                                                  axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label, framerate,
                                                  "cubed_sphere_aquaplanet_ζ_latitude-depth_section_$panel_index";
                                                  use_prettytimes = true, prettytimes = prettytimes)

        title = "Surface elevation"
        cbar_label = "surface elevation (m)"
        create_single_line_or_scatter_plot_animation(resolution, plot_type_1D,
                                                     latitude_at_specific_longitude_through_panel_center[:, index],
                                                     η_at_specific_longitude_through_panel_center[:, :, 1, index],
                                                     axis_kwargs_ssh, title, plot_kwargs, framerate,
                                                     "cubed_sphere_aquaplanet_η_vs_latitude_$panel_index";
                                                     use_prettytimes = true, prettytimes = prettytimes,
                                                     use_symmetric_range = false, tight_x_axis = true)

        title = "Buoyancy"
        cbar_label = "buoyancy (m s⁻²)"
        create_heat_map_or_contour_plot_animation(resolution, plot_type_2D,
                                                  latitude_at_specific_longitude_through_panel_center[:, index],
                                                  depths/1000, b_at_specific_longitude_through_panel_center[:, :, :,
                                                                                                            index],
                                                  axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label, framerate,
                                                  "cubed_sphere_aquaplanet_b_latitude-depth_section_$panel_index";
                                                  use_prettytimes = true, prettytimes = prettytimes)
    end

    u_colorrange = specify_colorrange_timeseries(grid, u_timeseries; common_kwargs_geo_colorrange...)
    copy_to_center_field(x_timeseries, u_timeseries)
    geo_heatlatlon_visualization_animation(grid, x_timeseries, "cc", prettytimes, "Zonal velocity",
                                           "cubed_sphere_aquaplanet_u_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "zonal velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = u_colorrange, framerate = framerate)

    v_colorrange = specify_colorrange_timeseries(grid, v_timeseries; common_kwargs_geo_colorrange...)
    copy_to_center_field(x_timeseries, v_timeseries)
    geo_heatlatlon_visualization_animation(grid, x_timeseries, "cc", prettytimes, "Meridional velocity",
                                           "cubed_sphere_aquaplanet_v_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "meridional velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = v_colorrange, framerate = framerate)

    ζ_colorrange = specify_colorrange_timeseries(grid, ζ_timeseries; common_kwargs_geo_colorrange...)
    copy_to_center_field(x_timeseries, ζ_timeseries)
    geo_heatlatlon_visualization_animation(grid, x_timeseries, "cc", prettytimes, "Relative vorticity",
                                           "cubed_sphere_aquaplanet_ζ_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "relative vorticity (s⁻¹)", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange, framerate = framerate)

    η_colorrange = specify_colorrange_timeseries(grid, η_timeseries; ssh = true)
    geo_heatlatlon_visualization_animation(grid, η_timeseries, "cc", prettytimes, "Surface elevation",
                                           "cubed_sphere_aquaplanet_η_geo_heatlatlon_animation"; ssh = true,
                                           cbar_label = "surface elevation (m)", specify_plot_limits = true,
                                           plot_limits = η_colorrange, framerate = framerate)

    b_colorrange = specify_colorrange_timeseries(grid, b_timeseries; common_kwargs_geo_colorrange_b...)
    geo_heatlatlon_visualization_animation(grid, b_timeseries, "cc", prettytimes, "Buoyancy",
                                           "cubed_sphere_aquaplanet_b_geo_heatlatlon_animation"; k = b_index,
                                           cbar_label = "buoyancy (m s⁻²)", specify_plot_limits = true,
                                           plot_limits = b_colorrange, framerate = framerate)
end
