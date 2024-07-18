using Adapt
using CUDA
using JLD2
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Grids: node, λnode, φnode, halo_size, total_size
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

Lz = 3000
h_b = 0.25 * Lz
h_νz_κz = 100

Nx, Ny, Nz = 360, 360, 30
Nhalo = 4

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
Ld = (2/f₀ * sqrt(my_parameters.h_b * my_parameters.Δ/(1 - exp(-my_parameters.Lz/my_parameters.h_b)))
      * (1 - exp(-my_parameters.Lz/(2my_parameters.h_b))))
print(
"For an initial buoyancy profile decaying exponentially with depth, the Rossby radius of deformation is $Ld m.\n")
Nx_min = ceil(Int, 2π * radius/(4Ld))
print("The minimum number of grid points in each direction of the cubed sphere panels required to resolve this " *
      "Rossby radius of deformation is $(Nx_min).\n")

arch = CPU()
underlying_grid = ConformalCubedSphereGrid(arch;
                                           panel_size = (Nx, Ny, Nz),
                                           z = geometric_z_faces(my_parameters),
                                           horizontal_direction_halo = Nhalo,
                                           radius,
                                           partition = CubedSpherePartition(; R = 1))

Δλ = 1
Δφ = 1

@inline function double_drake_depth(λ, φ)
    if ((-40 < φ < 75) & ((-Δλ < λ ≤ 0) | (90 ≤ λ < (90 + Δλ)))) | ((75 < φ < (75 + Δφ)) & (-Δλ < λ ≤ (90 + Δλ)))
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

import Oceananigans: on_architecture
cpu_grid = on_architecture(CPU(), grid)  

location = (Face(), Center(), Center())
@apply_regionally zonal_wind_stress_fc = wind_stress(cpu_grid, location, my_parameters)
@apply_regionally zonal_wind_stress_fc = on_architecture(arch, zonal_wind_stress_fc)

location = (Center(), Face(), Center())
@apply_regionally zonal_wind_stress_cf = wind_stress(cpu_grid, location, my_parameters)
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
top_stress_x = FluxBoundaryCondition(u_stress; discrete_form = true)
top_stress_y = FluxBoundaryCondition(v_stress; discrete_form = true)

u_bcs = FieldBoundaryConditions(bottom = u_bot_bc, top = top_stress_x)
v_bcs = FieldBoundaryConditions(bottom = v_bot_bc, top = top_stress_y)

my_buoyancy_parameters = (; Δ = my_parameters.Δ, h = my_parameters.h_b, Lz = my_parameters.Lz,
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
substeps           = 50
free_surface       = SplitExplicitFreeSurface(grid; substeps, extended_halos = false)

νh = 5e+3
κh = 1e+2 
horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh, κ=κh) # Laplacian viscosity and diffusivity

νz_surface = 1e-3
νz_bottom = 1e-4

struct MyVerticalViscosity{FT} <: Function
    Lz  :: FT
    h   :: FT
    νzs :: FT
    νzb :: FT
end

using Adapt

Adapt.adapt_structure(to, ν::MyVerticalViscosity) = MyVerticalViscosity(Adapt.adapt(to, ν.Lz),  Adapt.adapt(to, ν.h),
                                                                        Adapt.adapt(to, ν.νzs), Adapt.adapt(to, ν.νzb))

@inline (ν::MyVerticalViscosity)(x, y, z, t) = ν.νzb + (ν.νzs - ν.νzb) * exponential_profile_in_z(z, ν.Lz, ν.h)

νz = MyVerticalViscosity(float(Lz), float(h_νz_κz), νz_surface, νz_bottom)

κz_surface = 2e-4
κz_bottom = 2e-5

κz = MyVerticalViscosity(float(Lz), float(h_νz_κz), κz_surface, κz_bottom)

vertical_diffusivity  = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = νz, κ = κz)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization(),
                                                                convective_κz = 1.0)

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
                                     * exponential_profile_in_z(z, my_parameters.Lz, my_parameters.h_b))
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

compute_vorticity!(grid, model.velocities.u, model.velocities.v, ζ)

# Compute actual and reconstructed wind stress.
location = (Center(), Center(), Center())
@apply_regionally zonal_wind_stress_cc = wind_stress(cpu_grid, location, my_parameters)
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
uᵢ = on_architecture(CPU(), deepcopy(model.velocities.u))
vᵢ = on_architecture(CPU(), deepcopy(model.velocities.v))
ζᵢ = on_architecture(CPU(), deepcopy(ζ))
bᵢ = on_architecture(CPU(), deepcopy(model.tracers.b))

include("cubed_sphere_visualization.jl")

latitude = extract_latitude(cpu_grid)
cos_θ, sin_θ = calculate_sines_and_cosines_of_cubed_sphere_grid_angles(cpu_grid, "cc")

cos_θ_at_specific_longitude_through_panel_center    = zeros(2*Nx, 4);
sin_θ_at_specific_longitude_through_panel_center    = zeros(2*Nx, 4);
latitude_at_specific_longitude_through_panel_center = zeros(2*Nx, 4);

for (index, panel_index) in enumerate([1])
    cos_θ_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(cpu_grid, cos_θ, panel_index))
    sin_θ_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(cpu_grid, sin_θ, panel_index))
    latitude_at_specific_longitude_through_panel_center[:, index] = (
    extract_scalar_at_specific_longitude_through_panel_center(cpu_grid, latitude, panel_index))
end

depths = cpu_grid[1].zᵃᵃᶜ[1:Nz]
depths_f = cpu_grid[1].zᵃᵃᶠ[1:Nz+1]

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
axis_kwargs_Ld = (; axis_kwargs..., ylabel = "Deformation radius (m)")
axis_kwargs_ssh = (; axis_kwargs..., ylabel = "Surface elevation (m)")
contourlevels = 50
cbar_kwargs = (labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
common_kwargs = (; consider_all_levels = false)
common_kwargs_Ld = (consider_all_levels = false, use_symmetric_colorrange = false)
b_index = round(Int, Nz/2)
w_index = 6
common_kwargs_geo_colorrange = (consider_all_levels = false, levels = Nz:Nz)
common_kwargs_geo_colorrange_b = (consider_all_levels = false, levels = b_index:b_index)
common_kwargs_geo_τ = (consider_all_levels = false, levels = 1:1)
common_kwargs_geo_Ld = (consider_all_levels = false, levels = 1:1, use_symmetric_colorrange = false)
common_kwargs_geo = (consider_all_levels = false, k = Nz)
common_kwargs_geo_b = (consider_all_levels = false, k = b_index)
common_kwargs_geo_w = (consider_all_levels = false, k = w_index)
Ld_max = 100e3

@inline _deformation_radius(i, j, k, grid, C, buoyancy, coriolis) = sqrt(max(0, ∂z_b(i, j, k, grid, buoyancy, C))) / π /
                                                                         abs(ℑxyᶜᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis))

φ_max_b = 75

@kernel function _calculate_deformation_radius!(Ld, grid, tracers, buoyancy, coriolis)
    i, j = @index(Global, NTuple)

    @inbounds begin
        Ld[i, j, 1] = 0
        @unroll for k in 1:grid.Nz
            Ld[i, j, 1] += Δzᶜᶜᶠ(i, j, k, grid) * _deformation_radius(i, j, k, grid, tracers, buoyancy, coriolis)
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
buoyancy = model.buoyancy
tracers = model.tracers
set!(tracers.b, bᵢ)
coriolis = model.coriolis

@apply_regionally launch!(arch, grid, :xy, _calculate_deformation_radius!, Ldᵢ, grid, tracers, buoyancy, coriolis)
Ldᵢ_minimum = minimum(Ldᵢ)
@apply_regionally launch!(arch, grid, :xy, _truncate_deformation_radius!, Ldᵢ, grid, Ldᵢ_minimum)
Ldᵢ_at_specific_longitude_through_panel_center = zeros(2*Nx, 4);

plot_initial_field = true
if plot_initial_field
    fig = panel_wise_visualization(cpu_grid, on_architecture(CPU(), τ_x); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress.png", fig)

    fig = panel_wise_visualization(cpu_grid, on_architecture(CPU(), τ_x_r); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed.png", fig)

    fig = panel_wise_visualization(cpu_grid, on_architecture(CPU(), τ_y_r); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed.png", fig)

    title = "Zonal wind stress"
    fig = geo_heatlatlon_visualization(cpu_grid, on_architecture(CPU(), τ_x), title; common_kwargs_geo_τ...,
                                       cbar_label = "zonal wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_zonal_wind_stress_geo_heatlatlon_plot.png", fig)

    title = "Reconstructed zonal wind stress"
    fig = geo_heatlatlon_visualization(cpu_grid, on_architecture(CPU(), τ_x_r), title; common_kwargs_geo_τ...,
                                       cbar_label = "zonal wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)

    title = "Reconstructed meridional wind stress"
    fig = geo_heatlatlon_visualization(cpu_grid, on_architecture(CPU(), τ_y_r), title; common_kwargs_geo_τ...,
                                       cbar_label = "meridional wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)

    if initialize_velocities_based_on_thermal_wind_balance
        uᵢ, vᵢ = orient_velocities_in_global_direction(cpu_grid, uᵢ, vᵢ, cos_θ, sin_θ; levels = 1:Nz)

        fig = panel_wise_visualization(cpu_grid, uᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_uᵢ.png", fig)

        fig = panel_wise_visualization(cpu_grid, vᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_vᵢ.png", fig)

        ζᵢ = interpolate_cubed_sphere_field_to_cell_centers(cpu_grid, ζᵢ, "ff"; levels = 1:Nz)

        fig = panel_wise_visualization(cpu_grid, ζᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_ζᵢ.png", fig)
        
        title = "Initial zonal velocity"
        fig = geo_heatlatlon_visualization(cpu_grid, uᵢ, title; common_kwargs_geo..., cbar_label = "zonal velocity (m s⁻¹)")
        save("cubed_sphere_aquaplanet_u_0.png", fig)

        title = "Initial meridional velocity"
        fig = geo_heatlatlon_visualization(cpu_grid, vᵢ, title; common_kwargs_geo...,
                                           cbar_label = "meridional velocity (m s⁻¹)")
        save("cubed_sphere_aquaplanet_v_0.png", fig)

        title = "Initial relative vorticity"
        fig = geo_heatlatlon_visualization(cpu_grid, ζᵢ, title; common_kwargs_geo...,
                                           cbar_label = "relative vorticity (s⁻¹)")
        save("cubed_sphere_aquaplanet_ζ_0.png", fig)

        index, panel_index = 1, 1
        
        uᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(cpu_grid, uᵢ, panel_index; levels = 1:Nz))
        vᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(cpu_grid, vᵢ, panel_index; levels = 1:Nz))
        ζᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(cpu_grid, ζᵢ, panel_index; levels = 1:Nz))

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

    fig = panel_wise_visualization(cpu_grid, bᵢ; k = b_index, common_kwargs...)
    save("cubed_sphere_aquaplanet_bᵢ.png", fig)
    
    fig = panel_wise_visualization(cpu_grid, on_architecture(CPU(), Ldᵢ); k = 1, common_kwargs_Ld...)
    save("cubed_sphere_aquaplanet_Ldᵢ.png", fig)
    
    title = "Initial buoyancy"
    fig = geo_heatlatlon_visualization(cpu_grid, bᵢ, title; common_kwargs_geo_b..., cbar_label = "buoyancy (m s⁻²)")
    save("cubed_sphere_aquaplanet_b_0.png", fig)
    
    title = "Deformation radius"
    fig = geo_heatlatlon_visualization(cpu_grid, on_architecture(CPU(), Ldᵢ), title; common_kwargs_geo_Ld...,
                                       cbar_label = "deformation radius (m)")
    save("cubed_sphere_aquaplanet_Ldᵢ_geo_heatlatlon_plot.png", fig)
    
    index, panel_index = 1, 1
    
    bᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
    extract_field_at_specific_longitude_through_panel_center(cpu_grid, bᵢ, panel_index; levels = 1:Nz))
    title = "Buoyancy"
    cbar_label = "buoyancy (m s⁻²)"
    create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                    latitude_at_specific_longitude_through_panel_center[:, index], depths/1000,
                                    bᵢ_at_specific_longitude_through_panel_center[:, :, index], axis_kwargs, title,
                                    contourlevels, cbar_kwargs, cbar_label,
                                    "cubed_sphere_aquaplanet_bᵢ_latitude-depth_section_$panel_index")
    
    Ldᵢ_at_specific_longitude_through_panel_center[:, index] = (
    extract_field_at_specific_longitude_through_panel_center(cpu_grid, Ldᵢ, panel_index; levels = 1:1))
    title = "Deformation radius"
    create_single_line_or_scatter_plot(resolution, plot_type_1D,
                                       latitude_at_specific_longitude_through_panel_center[:, index],
                                       log10.(Ldᵢ_at_specific_longitude_through_panel_center[:, index]), axis_kwargs_Ld,
                                       title, plot_kwargs, "cubed_sphere_aquaplanet_Ldᵢ_latitude_$panel_index";
                                       tight_x_axis = true)
end

iteration_id = 1036800

file_c = jldopen("cubed_sphere_aquaplanet_checkpointer_iteration$(iteration_id).jld2")

u_f = file_c["u/data"]
v_f = file_c["v/data"]

u_f_r, v_f_r = orient_velocities_in_global_direction(cpu_grid, u_f, v_f, cos_θ, sin_θ; levels = 1:Nz,
                                                     read_parent_field_data = true)

u_f = set_parent_field_data(cpu_grid, u_f, "fc"; levels = 1:Nz)
v_f = set_parent_field_data(cpu_grid, v_f, "cf"; levels = 1:Nz)

compute_vorticity!(cpu_grid, u_f, v_f, ζ)

ζ_f = interpolate_cubed_sphere_field_to_cell_centers(cpu_grid, ζ, "ff"; levels = 1:Nz)

w_f = file_c["w/data"]
w_f = set_parent_field_data(cpu_grid, w_f, "cc"; levels = 1:Nz+1)

η_f = file_c["η/data"]
η_f = set_parent_field_data(cpu_grid, η_f, "cc"; ssh = true)

b_f = file_c["b/data"]
b_f = set_parent_field_data(cpu_grid, b_f, "cc"; levels = 1:Nz)
set!(tracers.b, b_f)

Ld_f = Field((Center, Center, Nothing), grid)
@apply_regionally launch!(arch, grid, :xy, _calculate_deformation_radius!, Ld_f, grid, tracers, buoyancy, coriolis)
Ld_f_minimum = minimum(Ld_f)
@apply_regionally launch!(arch, grid, :xy, _truncate_deformation_radius!, Ld_f, grid, Ld_f_minimum)
Ld_f_at_specific_longitude_through_panel_center = zeros(2*Nx, 4);

Δt = 5minutes
simulation_time = iteration_id * Δt

title = "Zonal velocity after $(prettytime(simulation_time))"
fig = geo_heatlatlon_visualization(cpu_grid, u_f_r, title; common_kwargs_geo..., cbar_label = "zonal velocity (m s⁻¹)")
save("cubed_sphere_aquaplanet_u_f_geo_heatlatlon_plot_$iteration_id.png", fig)

title = "Meridional velocity after $(prettytime(simulation_time))"
fig = geo_heatlatlon_visualization(cpu_grid, v_f_r, title; common_kwargs_geo...,
                                   cbar_label = "meridional velocity (m s⁻¹)")
save("cubed_sphere_aquaplanet_v_f_geo_heatlatlon_plot_$iteration_id.png", fig)

title = "Relative vorticity after $(prettytime(simulation_time))"
fig = geo_heatlatlon_visualization(cpu_grid, ζ_f, title; common_kwargs_geo..., cbar_label = "relative vorticity (s⁻¹)",
                                   specify_plot_limits = true, plot_limits = (-1.25e-5, 1.25e-5))
save("cubed_sphere_aquaplanet_ζ_f_geo_heatlatlon_plot_$iteration_id.png", fig)

title = "Vertical velocity after $(prettytime(simulation_time))"
fig = geo_heatlatlon_visualization(cpu_grid, w_f, title; common_kwargs_geo_w...,
                                   cbar_label = "vertical velocity (m s⁻¹)", specify_plot_limits = true,
                                   plot_limits = (-5e-4, 5e-4))
save("cubed_sphere_aquaplanet_w_f_geo_heatlatlon_plot_$iteration_id.png", fig)

title = "Surface elevation after $(prettytime(simulation_time))"
fig = geo_heatlatlon_visualization(cpu_grid, η_f, title; ssh = true, cbar_label = "surface elevation (m)")
save("cubed_sphere_aquaplanet_η_f_geo_heatlatlon_plot_$iteration_id.png", fig)

title = "Buoyancy after $(prettytime(simulation_time))"
fig = geo_heatlatlon_visualization(cpu_grid, b_f, title; common_kwargs_geo_b..., cbar_label = "buoyancy (m s⁻²)",
                                   specify_plot_limits = true, plot_limits = (-0.055, 0.055))
save("cubed_sphere_aquaplanet_b_f_geo_heatlatlon_plot_$iteration_id.png", fig)

title = "Deformation radius after $(prettytime(simulation_time))"
fig = geo_heatlatlon_visualization(cpu_grid, Ld_f, title; common_kwargs_geo_Ld...,
                                   cbar_label = "deformation radius (m)")
save("cubed_sphere_aquaplanet_Ld_f_geo_heatlatlon_plot_$iteration_id.png", fig)

close(file_c)

u_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);
v_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);
ζ_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);
w_f_at_specific_longitude_through_panel_center  = zeros(2*Nx, Nz+1, 4);
η_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,    1, 4);
b_f_at_specific_longitude_through_panel_center  = zeros(2*Nx,   Nz, 4);

index, panel_index = 1, 1

u_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(cpu_grid, u_f_r, panel_index; levels = 1:Nz))

v_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(cpu_grid, v_f_r, panel_index; levels = 1:Nz))

ζ_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(cpu_grid, ζ_f, panel_index; levels = 1:Nz))

w_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(cpu_grid, w_f, panel_index; levels = 1:Nz+1))

η_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(cpu_grid, η_f, panel_index; levels = Nz+1:Nz+1))

b_f_at_specific_longitude_through_panel_center[:, :, index] = (
extract_field_at_specific_longitude_through_panel_center(cpu_grid, b_f, panel_index; levels = 1:Nz))

Ld_f_at_specific_longitude_through_panel_center[:, index] = (
extract_field_at_specific_longitude_through_panel_center(cpu_grid, Ld_f, panel_index; levels = 1:1))

title = "Zonal velocity after $(prettytime(simulation_time))"
cbar_label = "zonal velocity (m s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, u_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_u_f_latitude-depth_section_$panel_index")

title = "Meridional velocity after $(prettytime(simulation_time))"
cbar_label = "meridional velocity (m s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, v_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_v_f_latitude-depth_section_$panel_index")

title = "Relative vorticity after $(prettytime(simulation_time))"
cbar_label = "relative vorticity (s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, ζ_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_ζ_f_latitude-depth_section_$panel_index";
                                specify_plot_limits = true, plot_limits = (-1e-5, 1e-5))

title = "Vertical velocity after $(prettytime(simulation_time))"
cbar_label = "vertical velocity (s⁻¹)"
create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                latitude_at_specific_longitude_through_panel_center[:, index],
                                depths_f[2:Nz+1]/1000, w_f_at_specific_longitude_through_panel_center[:, 2:Nz+1, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_w_f_latitude-depth_section_$panel_index")

title = "Surface elevation after $(prettytime(simulation_time))"
create_single_line_or_scatter_plot(resolution, plot_type_1D,
                                   latitude_at_specific_longitude_through_panel_center[:, index],
                                   η_f_at_specific_longitude_through_panel_center[:, 1, index], axis_kwargs_ssh,
                                   title, plot_kwargs, "cubed_sphere_aquaplanet_η_f_latitude_$panel_index";
                                   tight_x_axis = true)
title = "Buoyancy after $(prettytime(simulation_time))"
cbar_label = "buoyancy (m s⁻²)"
create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                latitude_at_specific_longitude_through_panel_center[:, index],
                                depths/1000, b_f_at_specific_longitude_through_panel_center[:, :, index],
                                axis_kwargs, title, contourlevels, cbar_kwargs, cbar_label,
                                "cubed_sphere_aquaplanet_b_f_latitude-depth_section_$panel_index";
                                specify_plot_limits = true, plot_limits = (-0.055, 0.055))

title = "Deformation radius after $(prettytime(simulation_time))"
create_single_line_or_scatter_plot(resolution, plot_type_1D,
                                   (latitude_at_specific_longitude_through_panel_center[:, index]),
                                   log10.(Ld_f_at_specific_longitude_through_panel_center[:, index]), axis_kwargs_Ld,
                                   title, plot_kwargs, "cubed_sphere_aquaplanet_Ld_f_latitude_$panel_index";
                                   tight_x_axis = true)
