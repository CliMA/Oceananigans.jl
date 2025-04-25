using Adapt
using CUDA
using JLD2
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Grids: node, λnode, φnode, halo_size, total_size
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.MultiRegion: getregion, number_of_regions, fill_halo_regions!, Iterate
using Oceananigans.Operators
using Oceananigans.Operators: Δx, Δy
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.TurbulenceClosures
using Oceananigans.Units
using Oceananigans.Utils
using Printf

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

function hyperbolic_tangential_z_faces(Lz)
    Δz_tolerance = 1e-2
    N = 20
    b = (atanh(1 - Δz_tolerance) - atanh(-1 + Δz_tolerance))/(N-1)
    k₀ = 1 - atanh(-1 + Δz_tolerance)/b
    a = 45
    c = a + 10
    Δz = zeros(N)
    for k in 1:N
        Δz[k] = a * tanh(b*(k - k₀)) + c
    end
    Nz₁ = 10
    Nz₂ = N
    Nz₃ = trunc(Int, (Lz - sum(Δz) - 100) ÷ 100)
    Nz = Nz₁ + Nz₂ + Nz₃
    z_faces = zeros(Nz+1)
    for k in 1:Nz₁+1
        z_faces[k] = 10(k - 1)
    end
    for k in Nz₁+2:Nz₁+Nz₂+1
        z_faces[k] = z_faces[k-1] + Δz[k-Nz₁-1]
    end
    for k in Nz₁+Nz₂+2:Nz+1
        z_faces[k] = z_faces[k-1] + 100
    end
    z_faces = reverse(-z_faces)
    return z_faces
end

function custom_z_faces()
    z_faces = [-3000, -2900, -2800, -2700, -2600, -2500, -2400, -2300, -2200, -2100, -2000, -1900, -1800, -1700, -1600,
               -1500, -1400, -1300, -1200, -1100, -1002, -904, -809, -717, -629, -547, -472, -404, -345, -294, -252,
               -217, -189, -167, -149, -134, -122, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0]
    return z_faces
end

Lz = 3000
h_b = 0.2 * Lz
h_νz_κz = 100

Nx, Ny, Nz = 360, 360, 48
Nhalo = 6

ratio = 0.8

φ_max_τ = 70
φs = (-φ_max_τ, -45, -15, 0, 15, 45, φ_max_τ)
τs = (0, 0.2, -0.1, -0.02, -0.1, 0.2, 0)

my_parameters = (Lz          = Lz,
                 h_b         = h_b,
                 h_νz_κz     = h_νz_κz,
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
L_d = (2/f₀ * sqrt(my_parameters.h_b * my_parameters.Δ/(1 - exp(-my_parameters.Lz/my_parameters.h_b)))
       * (1 - exp(-my_parameters.Lz/(2my_parameters.h_b))))
print(
"For an initial buoyancy profile decaying exponentially with depth, the Rossby radius of deformation is $L_d m.\n")
Nx_min = ceil(Int, 2π * radius/(4L_d))
print("The minimum number of grid points in each direction of the cubed sphere panels required to resolve this " *
      "Rossby radius of deformation is $(Nx_min).\n")

arch = CPU()
underlying_grid = ConformalCubedSphereGrid(arch;
                                           panel_size = (Nx, Ny, Nz),
                                           z = hyperbolic_tangential_z_faces(Lz),
                                           horizontal_direction_halo = Nhalo,
                                           radius,
                                           non_uniform_conformal_mapping = true,
                                           partition = CubedSpherePartition(; R = 1))

import Oceananigans: on_architecture
underlying_grid_cpu = on_architecture(CPU(), underlying_grid)

Nc = Nx
Nc_mid = isodd(Nc) ? (Nc + 1)÷2 : Nc÷2

φ_min = -34
filtered_φ_indices = findall(x -> x < φ_min, underlying_grid_cpu[1].φᶜᶜᵃ[Nc_mid, :])
Nc_min = maximum(filtered_φ_indices)

@inline function double_drake_bottom_depth(region, Lz, Nc, Nc_mid)
    bottom_depth = -ones(Nc, Nc) * Lz
    if region == 3
        bottom_depth[Nc_mid:Nc_mid+1, 1:Nc_mid+1] .= 0
        bottom_depth[1:Nc_mid+1, Nc_mid:Nc_mid+1] .= 0
    end
    if isodd(Nc)
        if region == 1
            bottom_depth[Nc_mid-1:Nc_mid, Nc_min:Nc] .= 0
        elseif region == 2
            bottom_depth[Nc_mid:Nc_mid+1, Nc_min:Nc] .= 0
        end
    else
        if region == 1 || region == 2
            bottom_depth[Nc_mid:Nc_mid+1, Nc_min:Nc] .= 0
        end
    end
    return bottom_depth
end

multi_region = MultiRegionObject((1, 2, 3, 4, 5, 6))
@apply_regionally bottom_depth = double_drake_bottom_depth(multi_region, Lz, Nc, Nc_mid)

double_drake = false
grid = double_drake ? ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_depth)) : underlying_grid;

grid_cpu = on_architecture(CPU(), grid)

include("cubed_sphere_visualization.jl")

if double_drake
    fig = panel_wise_visualization(grid_cpu, grid_cpu.immersed_boundary.bottom_height; k = 1,
                                   consider_all_levels = false)
    save("cubed_sphere_aquaplanet_bottom_depth.png", fig)

    title = "Bottom depth"
    fig = geo_heatlatlon_visualization(grid_cpu, grid_cpu.immersed_boundary.bottom_height, title;
                                       consider_all_levels = false, cbar_label = "bottom depth (m)")
    save("cubed_sphere_aquaplanet_bottom_depth_geo_heatlatlon_plot.png", fig)
end

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

# Specify the wind stress as a function of latitude, φ.
@inline function wind_stress(grid, location, p)
    stress = zeros(grid.Nx, grid.Ny)
    
    for j in 1:grid.Ny, i in 1:grid.Nx
        φ = φnode(i, j, 1, grid, location...)

        if abs(φ) > p.φ_max_τ
            stress[i, j] = 0
        else
            φ_index = sum(φ .> p.φs) + 1

            φ₁ = p.φs[φ_index-1]
            φ₂ = p.φs[φ_index]
            τ₁ = p.τs[φ_index-1]
            τ₂ = p.τs[φ_index]

            stress[i, j] = -cubic_interpolate(φ, φ₁, φ₂, τ₁, τ₂) / p.ρ₀
        end     
    end
    
    return stress
end

location = (Face(), Center(), Center())
@apply_regionally zonal_wind_stress_fc = wind_stress(grid_cpu, location, my_parameters)
@apply_regionally zonal_wind_stress_fc = on_architecture(arch, zonal_wind_stress_fc)

location = (Center(), Face(), Center())
@apply_regionally zonal_wind_stress_cf = wind_stress(grid_cpu, location, my_parameters)
@apply_regionally zonal_wind_stress_cf = on_architecture(arch, zonal_wind_stress_cf)

struct WindStressBCX{C} <: Function
    stress :: C
end

struct WindStressBCY{C} <: Function
    stress :: C
end

on_architecture(to, τ::WindStressBCX) = WindStressBCX(on_architecture(to, τ.stress))
on_architecture(to, τ::WindStressBCY) = WindStressBCY(on_architecture(to, τ.stress))

@inline function (τ::WindStressBCX)(i, j, grid, clock, fields)
    @inbounds τₓ = τ.stress[i, j] # Here τₓ is the zonal wind stress on a latitude-longitude grid.

    # Now, calculate the cosine of the angle with respect to the geographic north, and use it to determine the component
    # of τₓ in the local x direction of the cubed sphere panel.
    
    φᶠᶠᵃ_i_jp1 = φnode(i, j+1, 1, grid,   Face(),   Face(), Center())
    φᶠᶠᵃ_i_j   = φnode(i,   j, 1, grid,   Face(),   Face(), Center())
    Δyᶠᶜᵃ_i_j  =    Δy(i,   j, 1, grid,   Face(), Center(), Center())

    u_Pseudo = deg2rad(φᶠᶠᵃ_i_jp1 - φᶠᶠᵃ_i_j)/Δyᶠᶜᵃ_i_j

    φᶜᶜᵃ_i_j   = φnode(i,   j, 1, grid, Center(), Center(), Center())
    φᶜᶜᵃ_im1_j = φnode(i-1, j, 1, grid, Center(), Center(), Center())
    Δxᶠᶜᵃ_i_j  =    Δx(i,   j, 1, grid,   Face(), Center(), Center())

    v_Pseudo = -deg2rad(φᶜᶜᵃ_i_j - φᶜᶜᵃ_im1_j)/Δxᶠᶜᵃ_i_j

    cos_θ = u_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)

    τₓ_x = τₓ * cos_θ

    return τₓ_x
end

@inline function (τ::WindStressBCY)(i, j, grid, clock, fields)
    @inbounds τₓ = τ.stress[i, j] # Here τₓ is the zonal wind stress on a latitude-longitude grid.
    
    # Now, calculate the sine of the angle with respect to the geographic north, and use it to determine the component
    # of τₓ in the local y direction of the cubed sphere panel.

    φᶜᶜᵃ_i_j   = φnode(i,   j, 1, grid, Center(), Center(), Center())
    φᶜᶜᵃ_i_jm1 = φnode(i, j-1, 1, grid, Center(), Center(), Center())
    Δyᶜᶠᵃ_i_j  =    Δy(i,   j, 1, grid, Center(),   Face(), Center())

    u_Pseudo = deg2rad(φᶜᶜᵃ_i_j - φᶜᶜᵃ_i_jm1)/Δyᶜᶠᵃ_i_j

    φᶠᶠᵃ_ip1_j = φnode(i+1, j, 1, grid,   Face(),   Face(), Center())
    φᶠᶠᵃ_i_j   = φnode(i,   j, 1, grid,   Face(),   Face(), Center())
    Δxᶜᶠᵃ_i_j  =    Δx(i,   j, 1, grid, Center(),   Face(), Center())

    v_Pseudo = -deg2rad(φᶠᶠᵃ_ip1_j - φᶠᶠᵃ_i_j)/Δxᶜᶠᵃ_i_j

    sin_θ = v_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)

    τₓ_y = τₓ * sin_θ

    return τₓ_y
end

u_stress = WindStressBCX(zonal_wind_stress_fc)
v_stress = WindStressBCY(zonal_wind_stress_cf)

import Oceananigans.Utils: getregion, _getregion

@inline getregion(τ::WindStressBCX, i)  = WindStressBCX(_getregion(τ.stress, i))
@inline getregion(τ::WindStressBCY, i)  = WindStressBCY(_getregion(τ.stress, i))

@inline _getregion(τ::WindStressBCX, i) = WindStressBCX(getregion(τ.stress, i))
@inline _getregion(τ::WindStressBCY, i) = WindStressBCY(getregion(τ.stress, i))

@inline linear_profile_in_z(z, p)          = 1 + z/p.Lz
@inline exponential_profile_in_z(z, Lz, h) = (exp(z / h) - exp(-Lz / h)) / (1 - exp(-Lz / h))

@inline linear_profile_in_y(φ, p)        = 1 - abs(φ)/p.φ_max_b_lin
@inline parabolic_profile_in_y(φ, p)     = 1 - (φ/p.φ_max_b_par)^2
@inline cosine_profile_in_y(φ, p)        = 0.5(1 + cos(π * min(max(φ/p.φ_max_b_cos, -1), 1)))
@inline double_cosine_profile_in_y(φ, p) = (
0.5(1 + cos(π * min(max((deg2rad(abs(φ)) - π/4)/(deg2rad(p.φ_max_b_cos) - π/4), -1), 1))))

@inline function buoyancy_restoring(λ, φ, t, b, p)
    B = p.Δ * cosine_profile_in_y(φ, p)
    return p.𝓋 * (b - B)
end

extended_halos = true
coriolis = HydrostaticSphericalCoriolis()

my_buoyancy_parameters = (; Δ = my_parameters.Δ, h = my_parameters.h_b, Lz = my_parameters.Lz,
                            φ_max_b_lin = my_parameters.φ_max_b_lin, φ_max_b_par = my_parameters.φ_max_b_par,
                            φ_max_b_cos = my_parameters.φ_max_b_cos, 𝓋 = my_parameters.𝓋)
@inline initial_buoyancy(λ, φ, z) = (my_buoyancy_parameters.Δ * cosine_profile_in_y(φ, my_buoyancy_parameters)
                                     * exponential_profile_in_z(z, my_parameters.Lz, my_parameters.h_b))
# Specify the initial buoyancy profile to match the buoyancy restoring profile.
bᵢ = CenterField(grid)
set!(bᵢ, initial_buoyancy)

uᵢ = XFaceField(grid)
vᵢ = YFaceField(grid)

initialize_velocities_based_on_thermal_wind_balance = false
# If the above flag is set to true, meaning the velocities are initialized using thermal wind balance, set
# φ_max_b_cos within the range [70, 80], and specify the latitudinal variation in buoyancy as
# p.Δ * double_cosine_profile_in_y(φ, p) in both the initial buoyancy and the surface buoyancy restoring profiles.
if initialize_velocities_based_on_thermal_wind_balance
    fill_halo_regions!(bᵢ)

    Ω = coriolis.rotation_rate
    radius = grid.radius

    for region in 1:number_of_regions(grid), k in 1:Nz, j in 1:Ny, i in 1:Nx
        numerator = bᵢ[region][i, j, k] - bᵢ[region][i, j-1, k]
        denominator = -2Ω * sind(grid[region].φᶠᶜᵃ[i, j]) * grid[region].Δyᶠᶜᵃ[i, j]
        if k == 1
            Δz_below = grid[region].zᵃᵃᶜ[k] - grid[region].zᵃᵃᶠ[k]
            u_below = 0 # no slip boundary condition
        else
            Δz_below = grid[region].Δzᵃᵃᶠ[k]
            u_below = uᵢ[region][i, j, k-1]
        end
        uᵢ[region][i, j, k] = u_below + numerator/denominator * Δz_below
        numerator = bᵢ[region][i, j, k] - bᵢ[region][i-1, j, k]
        denominator = 2Ω * sind(grid[region].φᶜᶠᵃ[i, j]) * grid[region].Δxᶜᶠᵃ[i, j]
        if k == 1
            v_below = 0 # no slip boundary condition
        else
            v_below = vᵢ[region][i, j, k-1]
        end
        vᵢ[region][i, j, k] = v_below + numerator/denominator * Δz_below
    end

    fill_halo_regions!((uᵢ, vᵢ))
end

# Compute the initial vorticity.
ζ = Field{Face, Face, Center}(grid)

@kernel function _compute_vorticity!(grid, u, v, ζ)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

function compute_vorticity!(grid, u, v, ζ)
    offset = -1 .* halo_size(grid)

    fill_halo_regions!((u, v))

    @apply_regionally begin
        kernel_parameters = KernelParameters(total_size(ζ[1]), offset)
        launch!(arch, grid, kernel_parameters, _compute_vorticity!, grid, u, v, ζ)
    end
end

compute_vorticity!(grid, uᵢ, vᵢ, ζ)

# Compute actual and reconstructed wind stress.
location = (Center(), Center(), Center())
@apply_regionally zonal_wind_stress_cc = wind_stress(grid_cpu, location, my_parameters)
@apply_regionally zonal_wind_stress_cc = on_architecture(arch, zonal_wind_stress_cc)

struct ReconstructedWindStress{C} <: Function
    stress :: C
end

Adapt.adapt_structure(to, τ::ReconstructedWindStress) = ReconstructedWindStress(Adapt.adapt(to, τ.stress))

τ_x   = CenterField(grid, indices = (1:Nx, 1:Ny, 1:1)) # Specified zonal wind stress
τ_x_r = CenterField(grid, indices = (1:Nx, 1:Ny, 1:1)) # Reconstructed zonal wind stress
τ_y_r = CenterField(grid, indices = (1:Nx, 1:Ny, 1:1)) # Reconstructed meridional wind stress, expected to be zero

@kernel function _reconstruct_wind_stress!(grid, τₓ, τ_x, τ_x_r, τ_y_r)
    i, j = @index(Global, NTuple)

    τ_x[i, j, 1] = τₓ[i, j]

    φᶜᶠᵃ_i_jp1 = φnode(i, j+1, 1, grid, Center(),   Face(), Center())
    φᶜᶠᵃ_i_j   = φnode(i,   j, 1, grid, Center(),   Face(), Center())
    Δyᶜᶜᵃ_i_j  =    Δy(i,   j, 1, grid, Center(), Center(), Center())

    u_Pseudo = deg2rad(φᶜᶠᵃ_i_jp1 - φᶜᶠᵃ_i_j)/Δyᶜᶜᵃ_i_j

    φᶠᶜᵃ_ip1_j = φnode(i+1, j, 1, grid,   Face(), Center(), Center())
    φᶠᶜᵃ_i_j   = φnode(i,   j, 1, grid,   Face(), Center(), Center())
    Δxᶜᶜᵃ_i_j  =    Δx(i,   j, 1, grid, Center(), Center(), Center())

    v_Pseudo = -deg2rad(φᶠᶜᵃ_ip1_j - φᶠᶜᵃ_i_j)/Δxᶜᶜᵃ_i_j

    cos_θ = u_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)
    sin_θ = v_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)

    τₓ_x = τₓ[i, j] * cos_θ
    τₓ_y = τₓ[i, j] * sin_θ

    τ_x_r[i, j] = τₓ_x * cos_θ + τₓ_y * sin_θ
    τ_y_r[i, j] = τₓ_y * cos_θ - τₓ_x * sin_θ
end

@apply_regionally launch!(arch, grid, (Nx, Ny), _reconstruct_wind_stress!, grid, zonal_wind_stress_cc, τ_x, τ_x_r, τ_y_r)

# Plot wind stress and initial fields.
ζᵢ = on_architecture(CPU(), deepcopy(ζ))

latitude = extract_latitude(grid_cpu)
cos_θ, sin_θ = calculate_sines_and_cosines_of_cubed_sphere_grid_angles(grid_cpu, "cc")

cos_θ_at_specific_longitude_through_panel_center    = zeros(2*Nx, 4);
sin_θ_at_specific_longitude_through_panel_center    = zeros(2*Nx, 4);
latitude_at_specific_longitude_through_panel_center = zeros(2*Nx, 4);

for (index, panel_index) in enumerate([1])
    cos_θ_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(grid_cpu, cos_θ, panel_index))
    sin_θ_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(grid_cpu, sin_θ, panel_index))
    latitude_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(grid_cpu, latitude, panel_index))
end

depths = grid_cpu[1].zᵃᵃᶜ[1:Nz]
depths_f = grid_cpu[1].zᵃᵃᶠ[1:Nz+1]

uᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
vᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
ζᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);
bᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, Nz, 4);

resolution = (875, 750)
plot_type_1D = "line_plot"
plot_kwargs = (linewidth = 2, linecolor = :black, marker = :rect, markersize = 10)
plot_type_2D = "heat_map"
axis_kwargs = (xlabel = "Latitude (degrees)", ylabel = "Depth (km)", xlabelsize = 22.5, ylabelsize = 22.5,
               xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1,
               titlesize = 27.5, titlegap = 15, titlefont = :bold)
axis_kwargs_Ld = (axis_kwargs..., ylabel = "Deformation radius (m)")
axis_kwargs_η = (axis_kwargs..., ylabel = "Surface elevation (m)")
contourlevels = 50
cbar_kwargs = (labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
b_index = round(Int, Nz/2)
w_index = 6
common_kwargs = (; consider_all_levels = false)
common_kwargs_positive_scalar = (; consider_all_levels = false, use_symmetric_colorrange = false)
Ld_max = 100e3

use_grey_colormap = false
if use_grey_colormap
    common_kwargs = merge(common_kwargs, (specify_colormap = true, colormap = :greys))
    common_kwargs_positive_scalar = merge(common_kwargs_positive_scalar, (specify_colormap = true, colormap = :greys))
    common_kwargs_vertical_section = (; specify_colormap = true, colormap = :greys)
else
    common_kwargs_vertical_section = (; specify_colormap = false)
end
common_kwargs_η = common_kwargs_vertical_section

import Oceananigans.BuoyancyModels: ∂z_b
@inline ∂z_b(i, j, k, grid, buoyancy) = ∂zᶜᶜᶠ(i, j, k, grid, buoyancy)

@inline _deformation_radius(i, j, k, grid, buoyancy, coriolis) = (
sqrt(max(0, ∂z_b(i, j, k, grid, buoyancy))) / π / abs(ℑxyᶜᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis)))

φ_max_b = 75

@kernel function _calculate_deformation_radius!(Ld, grid, buoyancy, coriolis)
    i, j = @index(Global, NTuple)

    @inbounds begin
        Ld[i, j, 1] = 0
        @unroll for k in 1:grid.Nz
            Ld[i, j, 1] += Δzᶜᶜᶠ(i, j, k, grid) * _deformation_radius(i, j, k, grid, buoyancy, coriolis)
        end
    end

    Ld[i, j, 1] = min(Ld[i, j, 1], Ld_max)
    
    if abs(grid.φᶜᶜᵃ[i, j]) >= φ_max_b
        Ld[i, j, 1] = Ld_max
    end
end

@kernel function _truncate_deformation_radius!(Ld, grid, Ldᵢ_minimum)
    i, j = @index(Global, NTuple)
    
    if abs(grid.φᶜᶜᵃ[i, j]) >= φ_max_b
        Ld[i, j, 1] = Ldᵢ_minimum
    end
end

Ldᵢ = Field((Center, Center, Nothing), grid)

@apply_regionally launch!(arch, grid, :xy, _calculate_deformation_radius!, Ldᵢ, grid, bᵢ, coriolis)
Ldᵢ_minimum = minimum(Ldᵢ)
@apply_regionally launch!(arch, grid, :xy, _truncate_deformation_radius!, Ldᵢ, grid, Ldᵢ_minimum)
Ldᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, 4);

plot_initial_field = false
make_geo_heatlatlon_plots = true

if plot_initial_field
    fig = panel_wise_visualization(grid_cpu, on_architecture(CPU(), τ_x); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress.png", fig)

    fig = panel_wise_visualization(grid_cpu, on_architecture(CPU(), τ_x_r); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed.png", fig)

    fig = panel_wise_visualization(grid_cpu, on_architecture(CPU(), τ_y_r); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed.png", fig)

    if make_geo_heatlatlon_plots
        title = "Zonal wind stress"
        fig = geo_heatlatlon_visualization(grid_cpu, on_architecture(CPU(), τ_x), title; levels = 1:1, common_kwargs...,
                                           cbar_label = "zonal wind stress (N m⁻²)")
        save("cubed_sphere_aquaplanet_zonal_wind_stress_geo_heatlatlon_plot.png", fig)

        title = "Reconstructed zonal wind stress"
        fig = geo_heatlatlon_visualization(grid_cpu, on_architecture(CPU(), τ_x_r), title; levels = 1:1,
                                           common_kwargs..., cbar_label = "zonal wind stress (N m⁻²)")
        save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)

        title = "Reconstructed meridional wind stress"
        fig = geo_heatlatlon_visualization(grid_cpu, on_architecture(CPU(), τ_y_r), title; levels = 1:1,
                                           common_kwargs..., cbar_label = "meridional wind stress (N m⁻²)")
        save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)
    end

    if initialize_velocities_based_on_thermal_wind_balance
        uᵢ, vᵢ = orient_velocities_in_global_direction(grid_cpu, uᵢ, vᵢ, cos_θ, sin_θ; levels = 1:Nz)

        fig = panel_wise_visualization(grid_cpu, uᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_uᵢ.png", fig)

        fig = panel_wise_visualization(grid_cpu, vᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_vᵢ.png", fig)

        ζᵢ = interpolate_cubed_sphere_field_to_cell_centers(grid_cpu, ζᵢ, "ff"; levels = 1:Nz)

        fig = panel_wise_visualization(grid_cpu, ζᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_ζᵢ.png", fig)
        if make_geo_heatlatlon_plots
            title = "Initial zonal velocity"
            fig = geo_heatlatlon_visualization(grid_cpu, uᵢ, title; k = Nz, common_kwargs...,
                                               cbar_label = "zonal velocity (m s⁻¹)")
            save("cubed_sphere_aquaplanet_uᵢ_geo_heatlatlon_plot.png", fig)

            title = "Initial meridional velocity"
            fig = geo_heatlatlon_visualization(grid_cpu, vᵢ, title; k = Nz, common_kwargs...,
                                               cbar_label = "meridional velocity (m s⁻¹)")
            save("cubed_sphere_aquaplanet_vᵢ_geo_heatlatlon_plot.png", fig)

            title = "Initial relative vorticity"
            fig = geo_heatlatlon_visualization(grid_cpu, ζᵢ, title; k = Nz, common_kwargs...,
                                               cbar_label = "relative vorticity (s⁻¹)")
            save("cubed_sphere_aquaplanet_ζᵢ_geo_heatlatlon_plot.png", fig)
        end

        index, panel_index = 1, 1
        
        uᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, uᵢ, panel_index; levels = 1:Nz))
        vᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, vᵢ, panel_index; levels = 1:Nz))
        ζᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, ζᵢ, panel_index; levels = 1:Nz))

        title = "Zonal velocity"
        cbar_label = "zonal velocity (m s⁻¹)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index],
                                        depths/1000, uᵢ_at_specific_longitude_through_panel_center[:, :, index],
                                        axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_uᵢ_latitude-depth_section_$panel_index";
                                        common_kwargs_vertical_section...)
        title = "Meridional velocity"
        cbar_label = "meridional velocity (m s⁻¹)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index],
                                        depths/1000, vᵢ_at_specific_longitude_through_panel_center[:, :, index],
                                        axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_vᵢ_latitude-depth_section_$panel_index";
                                        common_kwargs_vertical_section...)
        title = "Relative vorticity"
        cbar_label = "relative vorticity (s⁻¹)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index],
                                        depths/1000, ζᵢ_at_specific_longitude_through_panel_center[:, :, index],
                                        axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_ζᵢ_latitude-depth_section_$panel_index";
                                        common_kwargs_vertical_section...)
    end

    fig = panel_wise_visualization(grid_cpu, bᵢ; k = b_index, common_kwargs...)
    save("cubed_sphere_aquaplanet_bᵢ.png", fig)
    
    fig = panel_wise_visualization(grid_cpu, on_architecture(CPU(), Ldᵢ); k = 1, common_kwargs_positive_scalar...)
    save("cubed_sphere_aquaplanet_Ldᵢ.png", fig)
    
    if make_geo_heatlatlon_plots
        title = "Initial buoyancy"
        fig = geo_heatlatlon_visualization(grid_cpu, bᵢ, title; k = b_index, common_kwargs...,
                                           cbar_label = "buoyancy (m s⁻²)")
        save("cubed_sphere_aquaplanet_bᵢ_geo_heatlatlon_plot.png", fig)

        title = "Deformation radius"
        fig = geo_heatlatlon_visualization(grid_cpu, on_architecture(CPU(), Ldᵢ), title; levels = 1:1,
                                           common_kwargs_positive_scalar..., cbar_label = "deformation radius (m)")
        save("cubed_sphere_aquaplanet_Ldᵢ_geo_heatlatlon_plot.png", fig)
    end

    index, panel_index = 1, 1
    
    bᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
    extract_field_at_specific_longitude_through_panel_center(grid_cpu, bᵢ, panel_index; levels = 1:Nz))
    title = "Buoyancy"
    cbar_label = "buoyancy (m s⁻²)"
    create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                    latitude_at_specific_longitude_through_panel_center[:, index], depths/1000,
                                    bᵢ_at_specific_longitude_through_panel_center[:, :, index], axis_kwargs, title,
                                    contourlevels, cbar_kwargs, cbar_label,
                                    "cubed_sphere_aquaplanet_bᵢ_latitude-depth_section_$panel_index";
                                    common_kwargs_vertical_section...)
    
    Ldᵢ_at_specific_longitude_through_panel_center[:, index] = (
    extract_field_at_specific_longitude_through_panel_center(grid_cpu, Ldᵢ, panel_index; levels = 1:1))
    title = "Deformation radius"
    create_single_line_or_scatter_plot(resolution, plot_type_1D,
                                       latitude_at_specific_longitude_through_panel_center[:, index],
                                       log10.(Ldᵢ_at_specific_longitude_through_panel_center[:, index]), axis_kwargs_Ld,
                                       title, plot_kwargs, "cubed_sphere_aquaplanet_Ldᵢ_latitude_$panel_index";
                                       specify_xticks = true, xticks = -90:30:90, tight_x_axis = true)
end

iteration_id = 2160000

file_c = jldopen("cubed_sphere_aquaplanet_checkpointer_iteration$(iteration_id).jld2")

u_f = file_c["u/data"]
v_f = file_c["v/data"]

u_f_r, v_f_r = orient_velocities_in_global_direction(grid_cpu, u_f, v_f, cos_θ, sin_θ; levels = 1:Nz,
                                                     read_parent_field_data = true)

u_f = set_parent_field_data(grid_cpu, u_f, "fc"; levels = 1:Nz)
v_f = set_parent_field_data(grid_cpu, v_f, "cf"; levels = 1:Nz)

compute_vorticity!(grid_cpu, u_f, v_f, ζ)

ζ_f = interpolate_cubed_sphere_field_to_cell_centers(grid_cpu, ζ, "ff"; levels = 1:Nz)

w_f = file_c["w/data"]
w_f = set_parent_field_data(grid_cpu, w_f, "cc"; levels = 1:Nz+1)

if extended_halos
    η_f_extended_halos = file_c["η/data"]
    Hc = grid.Hx
    Hc_extended = (size(η_f_extended_halos[1], 1) - Nc) ÷ 2
    η_f = Field((Center, Center, Center), grid; indices = (:, :, Nz+1:Nz+1))
    for region in 1:6, j in 1:Nc+2Hc, i in 1:Nc+2Hc
        η_f[region][i, j, Nz+1] = η_f_extended_halos[region][i+Hc_extended, j+Hc_extended, 1]
    end
else
    η_f = file_c["η/data"]
    η_f = set_parent_field_data(grid_cpu, η_f, "cc"; ssh = true)
end

b_f = file_c["b/data"]
b_f = set_parent_field_data(grid_cpu, b_f, "cc"; levels = 1:Nz)

Ld_f = Field((Center, Center, Nothing), grid)
@apply_regionally launch!(arch, grid, :xy, _calculate_deformation_radius!, Ld_f, grid, b_f, coriolis)
Ld_f_minimum = minimum(Ld_f)
@apply_regionally launch!(arch, grid, :xy, _truncate_deformation_radius!, Ld_f, grid, Ld_f_minimum)
Ld_f_at_specific_longitude_through_panel_center = zeros(2*Nx, 4);

Δt = 5minutes
simulation_time = iteration_id * Δt
specify_plot_limits = true
specify_η_limits = false
specify_b_limits = false

u_limits = (-0.75, 0.75)
v_limits = (-0.25, 0.25)
ζ_limits = (-5e-6, 5e-6)
w_limits = (-5e-5, 5e-5)
η_limits = (-10, 10)
b_limits = (-0.0325, 0.0325)

fig = panel_wise_visualization(grid_cpu, u_f_r; k = Nz, common_kwargs..., specify_plot_limits = specify_plot_limits,
                               plot_limits = u_limits)
save("cubed_sphere_aquaplanet_u_f_$iteration_id.png", fig)

fig = panel_wise_visualization(grid_cpu, v_f_r; k = Nz, common_kwargs..., specify_plot_limits = specify_plot_limits,
                               plot_limits = v_limits)
save("cubed_sphere_aquaplanet_v_f_$iteration_id.png", fig)

fig = panel_wise_visualization(grid_cpu, ζ_f; k = Nz, common_kwargs..., specify_plot_limits = specify_plot_limits,
                               plot_limits = ζ_limits)
save("cubed_sphere_aquaplanet_ζ_f_$iteration_id.png", fig)

fig = panel_wise_visualization(grid_cpu, w_f; k = w_index, common_kwargs..., specify_plot_limits = specify_plot_limits,
                               plot_limits = w_limits)
save("cubed_sphere_aquaplanet_w_f_$iteration_id.png", fig)

fig = panel_wise_visualization(grid_cpu, η_f; ssh = true, common_kwargs_η..., specify_plot_limits = specify_η_limits,
                               plot_limits = η_limits)
save("cubed_sphere_aquaplanet_η_f_$iteration_id.png", fig)

fig = panel_wise_visualization(grid_cpu, b_f; k = b_index, common_kwargs..., specify_plot_limits = specify_b_limits,
                               plot_limits = b_limits)
save("cubed_sphere_aquaplanet_b_f_$iteration_id.png", fig)

if make_geo_heatlatlon_plots
    title = "Zonal velocity after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid_cpu, u_f_r, title; k = Nz, common_kwargs...,
                                       cbar_label = "zonal velocity (m s⁻¹)", specify_plot_limits = specify_plot_limits,
                                       plot_limits = u_limits)
    save("cubed_sphere_aquaplanet_u_f_geo_heatlatlon_plot_$iteration_id.png", fig)

    title = "Meridional velocity after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid_cpu, v_f_r, title; k = Nz, common_kwargs...,
                                       cbar_label = "meridional velocity (m s⁻¹)",
                                       specify_plot_limits = specify_plot_limits, plot_limits = v_limits)
    save("cubed_sphere_aquaplanet_v_f_geo_heatlatlon_plot_$iteration_id.png", fig)

    title = "Relative vorticity after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid_cpu, ζ_f, title; k = Nz, common_kwargs...,
                                       cbar_label = "relative vorticity (s⁻¹)",
                                       specify_plot_limits = specify_plot_limits, plot_limits = ζ_limits)
    save("cubed_sphere_aquaplanet_ζ_f_geo_heatlatlon_plot_$iteration_id.png", fig)

    title = "Vertical velocity after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid_cpu, w_f, title; k = w_index, common_kwargs...,
                                       cbar_label = "vertical velocity (m s⁻¹)",
                                       specify_plot_limits = specify_plot_limits, plot_limits = w_limits)
    save("cubed_sphere_aquaplanet_w_f_geo_heatlatlon_plot_$iteration_id.png", fig)

    title = "Surface elevation after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid_cpu, η_f, title; ssh = true, common_kwargs_η...,
                                       cbar_label = "surface elevation (m)", specify_plot_limits = specify_η_limits,
                                       plot_limits = η_limits)
    save("cubed_sphere_aquaplanet_η_f_geo_heatlatlon_plot_$iteration_id.png", fig)

    title = "Buoyancy after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid_cpu, b_f, title; k = b_index, common_kwargs...,
                                       cbar_label = "buoyancy (m s⁻²)", specify_plot_limits = specify_b_limits,
                                       plot_limits = b_limits)
    save("cubed_sphere_aquaplanet_b_f_geo_heatlatlon_plot_$iteration_id.png", fig)

    title = "Deformation radius after $(prettytime(simulation_time))"
    fig = geo_heatlatlon_visualization(grid_cpu, Ld_f, title; levels = 1:1, common_kwargs_positive_scalar...,
                                       cbar_label = "deformation radius (m)")
    save("cubed_sphere_aquaplanet_Ld_f_geo_heatlatlon_plot_$iteration_id.png", fig)
end

close(file_c)

u_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);
v_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);
ζ_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);
w_f_at_specific_longitude_through_panel_center  = zeros(2*Nx, Nz+1, 4);
η_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,    1, 4);
b_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);

index, panel_index = 1, 1

u_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(grid_cpu, u_f_r, panel_index; levels = 1:Nz))

v_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(grid_cpu, v_f_r, panel_index; levels = 1:Nz))

ζ_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(grid_cpu, ζ_f, panel_index; levels = 1:Nz))

w_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(grid_cpu, w_f, panel_index; levels = 1:Nz+1))

η_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(grid_cpu, η_f, panel_index; levels = Nz+1:Nz+1))

b_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(grid_cpu, b_f, panel_index; levels = 1:Nz))

Ld_f_at_specific_longitude_through_panel_center[:, index] = (
extract_field_at_specific_longitude_through_panel_center(grid_cpu, Ld_f, panel_index; levels = 1:1))

title = "Zonal velocity after $(prettytime(simulation_time))"
cbar_label = "zonal velocity (m s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D, latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, u_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_u_f_latitude-depth_section_$(panel_index)_$(iteration_id)";
                                specify_plot_limits = false, plot_limits = u_limits, common_kwargs_vertical_section...)

title = "Meridional velocity after $(prettytime(simulation_time))"
cbar_label = "meridional velocity (m s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D, latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, v_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_v_f_latitude-depth_section_$(panel_index)_$(iteration_id)";
                                specify_plot_limits = false, plot_limits = v_limits, common_kwargs_vertical_section...)

title = "Relative vorticity after $(prettytime(simulation_time))"
cbar_label = "relative vorticity (s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D, latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, ζ_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_ζ_f_latitude-depth_section_$(panel_index)_$(iteration_id)";
                                specify_plot_limits = false, plot_limits = ζ_limits, common_kwargs_vertical_section...)

title = "Vertical velocity after $(prettytime(simulation_time))"
cbar_label = "vertical velocity (s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D, latitude_at_specific_longitude_through_panel_center[:, index],
                                depths_f[2:Nz+1]/1000, w_f_at_specific_longitude_through_panel_center[:, 2:Nz+1, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_w_f_latitude-depth_section_$(panel_index)_$(iteration_id)";
                                specify_plot_limits = false, plot_limits = w_limits, common_kwargs_vertical_section...)

title = "Surface elevation after $(prettytime(simulation_time))"
create_single_line_or_scatter_plot(resolution, plot_type_1D,
                                   latitude_at_specific_longitude_through_panel_center[:, index],
                                   η_f_at_specific_longitude_through_panel_center[:, 1, index], axis_kwargs_η,
                                   title, plot_kwargs,
                                   "cubed_sphere_aquaplanet_η_f_latitude_$(panel_index)_$(iteration_id)";
                                   tight_x_axis = true, specify_y_limits = false, y_limits = η_limits)

title = "Buoyancy after $(prettytime(simulation_time))"
cbar_label = "buoyancy (m s⁻²)"
create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, b_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_b_f_latitude-depth_section_$(panel_index)_$(iteration_id)";
                                specify_plot_limits = false, plot_limits = b_limits, common_kwargs_vertical_section...)

title = "Deformation radius after $(prettytime(simulation_time))"
create_single_line_or_scatter_plot(resolution, plot_type_1D,
                                   (latitude_at_specific_longitude_through_panel_center[:, index]),
                                   log10.(Ld_f_at_specific_longitude_through_panel_center[:, index]), axis_kwargs_Ld,
                                   title, plot_kwargs,
                                   "cubed_sphere_aquaplanet_Ld_f_latitude_$(panel_index)_$(iteration_id)";
                                   tight_x_axis = true)

if !isdir("cubed_sphere_aquaplanet_checkpointer_iteration$(iteration_id)")
    mkdir("cubed_sphere_aquaplanet_checkpointer_iteration$(iteration_id)")
end

# List all files with the .png extension
png_files = filter(x -> endswith(x, ".png"), readdir())

# Move each .png file to the "temp" directory
for file in png_files
    mv(file, joinpath("cubed_sphere_aquaplanet_checkpointer_iteration$(iteration_id)", file); force=true)
end

file = "cubed_sphere_aquaplanet_checkpointer_iteration$(iteration_id).jld2"
mv(file, joinpath("cubed_sphere_aquaplanet_checkpointer_iteration$(iteration_id)", file); force=true)
