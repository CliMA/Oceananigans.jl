using Adapt
using CUDA
using JLD2
using KernelAbstractions: @kernel, @index
using Oceananigans
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
                 φ_max_b_cos = 90,
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

arch = GPU()
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

momentum_advection = WENOVectorInvariant(vorticity_order=9)
tracer_advection   = WENO(order=9)
substeps           = 50
free_surface       = SplitExplicitFreeSurface(grid; substeps, extended_halos = true)

# Filter width squared, expressed as a harmonic mean of x and y spacings
@inline Δ²ᶜᶜᶜ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2
                                                      + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))

# Use a biharmonic diffusivity for momentum. Define the diffusivity function as gridsize^4 divided by the timescale.
@inline νhb(i, j, k, grid, lx, ly, lz, clock, fields, p) = Δ²ᶜᶜᶜ(i, j, k, grid, lx, ly, lz)^2 / p.λ_rts

horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = νhb, discrete_form = true,
                                                             vector_invariant_form = true,
                                                             parameters = (; λ_rts = 2days))

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
                                      closure = (vertical_diffusivity, convective_adjustment),
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
uᵢ = on_architecture(CPU(), deepcopy(model.velocities.u))
vᵢ = on_architecture(CPU(), deepcopy(model.velocities.v))
ζᵢ = on_architecture(CPU(), deepcopy(ζ))
bᵢ = on_architecture(CPU(), deepcopy(model.tracers.b))

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
    fig = panel_wise_visualization(grid_cpu, on_architecture(CPU(), τ_x); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress.png", fig)

    fig = panel_wise_visualization(grid_cpu, on_architecture(CPU(), τ_x_r); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed.png", fig)

    fig = panel_wise_visualization(grid_cpu, on_architecture(CPU(), τ_y_r); k = 1, common_kwargs...)
    save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed.png", fig)

    title = "Zonal wind stress"
    fig = geo_heatlatlon_visualization(grid_cpu, on_architecture(CPU(), τ_x), title; common_kwargs_geo_τ...,
                                       cbar_label = "zonal wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_zonal_wind_stress_geo_heatlatlon_plot.png", fig)

    title = "Reconstructed zonal wind stress"
    fig = geo_heatlatlon_visualization(grid_cpu, on_architecture(CPU(), τ_x_r), title; common_kwargs_geo_τ...,
                                       cbar_label = "zonal wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_zonal_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)

    title = "Reconstructed meridional wind stress"
    fig = geo_heatlatlon_visualization(grid_cpu, on_architecture(CPU(), τ_y_r), title; common_kwargs_geo_τ...,
                                       cbar_label = "meridional wind stress (N m⁻²)")
    save("cubed_sphere_aquaplanet_meridional_wind_stress_reconstructed_geo_heatlatlon_plot.png", fig)

    if initialize_velocities_based_on_thermal_wind_balance
        uᵢ, vᵢ = orient_velocities_in_global_direction(grid_cpu, uᵢ, vᵢ, cos_θ, sin_θ; levels = 1:Nz)

        fig = panel_wise_visualization(grid_cpu, uᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_uᵢ.png", fig)

        fig = panel_wise_visualization(grid_cpu, vᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_vᵢ.png", fig)

        ζᵢ = interpolate_cubed_sphere_field_to_cell_centers(grid_cpu, ζᵢ, "ff"; levels = 1:Nz)

        fig = panel_wise_visualization(grid_cpu, ζᵢ; k = Nz, common_kwargs...)
        save("cubed_sphere_aquaplanet_ζᵢ.png", fig)

        for (index, panel_index) in enumerate([1])
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
        fig = geo_heatlatlon_visualization(grid_cpu, uᵢ, title; common_kwargs_geo..., cbar_label = "zonal velocity (m s⁻¹)")
        save("cubed_sphere_aquaplanet_uᵢ_geo_heatlatlon_plot.png", fig)

        title = "Initial meridional velocity"
        fig = geo_heatlatlon_visualization(grid_cpu, vᵢ, title; common_kwargs_geo...,
                                           cbar_label = "meridional velocity (m s⁻¹)")
        save("cubed_sphere_aquaplanet_vᵢ_geo_heatlatlon_plot.png", fig)

        title = "Initial relative vorticity"
        fig = geo_heatlatlon_visualization(grid_cpu, ζᵢ, title; common_kwargs_geo...,
                                           cbar_label = "relative vorticity (s⁻¹)")
        save("cubed_sphere_aquaplanet_ζᵢ_geo_heatlatlon_plot.png", fig)
    end

    fig = panel_wise_visualization(grid_cpu, bᵢ; k = b_index, common_kwargs...)
    save("cubed_sphere_aquaplanet_bᵢ.png", fig)

    for (index, panel_index) in enumerate([1])
        bᵢ_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, bᵢ, panel_index; levels = 1:Nz))
        title = "Buoyancy"
        cbar_label = "buoyancy (m s⁻²)"
        create_heat_map_or_contour_plot(resolution, plot_type_2D,
                                        latitude_at_specific_longitude_through_panel_center[:, index], depths/1000,
                                        bᵢ_at_specific_longitude_through_panel_center[:, :, index], axis_kwargs, title,
                                        contourlevels, cbar_kwargs, cbar_label,
                                        "cubed_sphere_aquaplanet_bᵢ_latitude-depth_section_$panel_index")
    end

    title = "Initial buoyancy"
    fig = geo_heatlatlon_visualization(grid_cpu, bᵢ, title; common_kwargs_geo_b..., cbar_label = "buoyancy (m s⁻²)")
    save("cubed_sphere_aquaplanet_bᵢ_geo_heatlatlon_plot.png", fig)
end

#####
##### Simulation setup
#####

Δt = 12minutes

# Compute the minimum number of substeps required to satisfy the CFL condition for a given Courant number.
CUDA.@allowscalar min_spacing = filter(!iszero, grid.Δxᶠᶠᵃ) |> minimum
c = sqrt(model.free_surface.gravitational_acceleration * Lz)
CourantNumber = 0.7
min_substeps = ceil(Int, 2c * Δt / (CourantNumber * min_spacing))
print("The minimum number of substeps required to satisfy the CFL condition is $min_substeps.\n")

month = 30days
months = month
year = 365days
years = year
stop_time = 50years
save_fields_interval = 10days
save_surface_interval = 12hours
checkpointer_interval = 3months
# Note that n_frames = floor(Int, stop_time/save_fields_interval) + 1.

Ntime = round(Int, stop_time/Δt)

@info "Stop time = $(prettytime(stop_time))"
@info "Number of time steps = $Ntime"

simulation = Simulation(model; Δt, stop_time)

# Print a progress message.
progress_message_iteration_interval = 100

wall_time = [time_ns()]
function progress_message(sim) 
    @printf("Iteration: %04d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f, max|b|: %.3f, wall time: %s\n",
            iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u),
            maximum(abs, model.free_surface.η), maximum(abs, model.tracers.b), prettytime(1e-9 * (time_ns() - wall_time[1])))
    
    wall_time[1] = time_ns()
end

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(progress_message_iteration_interval))

#####
##### Build checkpointer and output writer
#####

filename_checkpointer = "cubed_sphere_aquaplanet_checkpointer"
simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(checkpointer_interval),
                                                        prefix = filename_checkpointer,
                                                        overwrite_existing = true)

outputs = fields(model)
filename_output_writer = "cubed_sphere_aquaplanet_output"
simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs;
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = filename_output_writer,
                                                      verbose = false,
                                                      overwrite_existing = true)

outputs = (u = model.velocities.u, v = model.velocities.v, b = model.tracers.b)
filename_output_writer = "cubed_sphere_aquaplanet_surface_output"
simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, outputs;
                                                              schedule = TimeInterval(save_surface_interval),
                                                              filename = filename_output_writer,
                                                              indices = (:, :, grid.Nz),
                                                              verbose = false,
                                                              overwrite_existing = true)

outputs = (; w = model.velocities.w, η = model.free_surface.η)
filename_output_writer = "cubed_sphere_aquaplanet_surface_output_w_η"
simulation.output_writers[:surface_w_η] = JLD2OutputWriter(model, outputs;
                                                           schedule = TimeInterval(save_surface_interval),
                                                           filename = filename_output_writer,
                                                           indices = (:, :, grid.Nz+1),
                                                           verbose = false,
                                                           overwrite_existing = true)

#####
##### Run simulation
#####

@info "Running the simulation..."

run!(simulation, pickup = false)

u_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "u"; architecture = CPU());
v_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "v"; architecture = CPU());
ζ_timeseries = Field[];
η_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "η"; architecture = CPU());
b_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "b"; architecture = CPU());

x_timeseries = FieldTimeSeries("cubed_sphere_aquaplanet_output.jld2", "b"; architecture = CPU());

n_frames = length(u_timeseries)

for i_frame in 1:n_frames
    compute_vorticity!(grid_cpu, u_timeseries[i_frame], v_timeseries[i_frame], ζ)
    push!(ζ_timeseries, deepcopy(ζ))
end

for i_frame in 1:n_frames
    u_frame, v_frame = (
    orient_velocities_in_global_direction(grid_cpu, u_timeseries[i_frame], v_timeseries[i_frame], cos_θ, sin_θ;
                                          levels = 1:Nz))
    ζ_frame = interpolate_cubed_sphere_field_to_cell_centers(grid_cpu, ζ_timeseries[i_frame], "ff"; levels = 1:Nz)
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
    fig = panel_wise_visualization(grid_cpu, u_timeseries[end]; k = Nz, common_kwargs...)
    save("cubed_sphere_aquaplanet_u_f.png", fig)

    fig = panel_wise_visualization(grid_cpu, v_timeseries[end]; k = Nz, common_kwargs...)
    save("cubed_sphere_aquaplanet_v_f.png", fig)

    fig = panel_wise_visualization(grid_cpu, ζ_timeseries[end]; k = Nz, common_kwargs...)
    save("cubed_sphere_aquaplanet_ζ_f.png", fig)

    fig = panel_wise_visualization(grid_cpu, η_timeseries[end]; ssh = true)
    save("cubed_sphere_aquaplanet_η_f.png", fig)

    fig = panel_wise_visualization(grid_cpu, b_timeseries[end]; k = b_index, common_kwargs...)
    save("cubed_sphere_aquaplanet_b_f.png", fig)

    for (index, panel_index) in enumerate([1])
        u_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, u_timeseries[end], panel_index;
                                                                 levels = 1:Nz))
        v_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, v_timeseries[end], panel_index;
                                                                 levels = 1:Nz))
        ζ_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, ζ_timeseries[end], panel_index;
                                                                 levels = 1:Nz))
        η_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, η_timeseries[end], panel_index;
                                                                 levels = Nz+1:Nz+1))
        b_f_at_specific_longitude_through_panel_center[:, :, index] = (
        extract_field_at_specific_longitude_through_panel_center(grid_cpu, b_timeseries[end], panel_index;
                                                                 levels = 1:Nz))
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

    u_colorrange = specify_colorrange_timeseries(grid_cpu, u_timeseries; common_kwargs_geo_colorrange...,
                                                 Δ = Δn_snapshots)
    v_colorrange = specify_colorrange_timeseries(grid_cpu, v_timeseries; common_kwargs_geo_colorrange...,
                                                 Δ = Δn_snapshots)
    ζ_colorrange = specify_colorrange_timeseries(grid_cpu, ζ_timeseries; common_kwargs_geo_colorrange...,
                                                 Δ = Δn_snapshots)
    η_colorrange = specify_colorrange_timeseries(grid_cpu, η_timeseries; ssh = true, Δ = Δn_snapshots)
    b_colorrange = specify_colorrange_timeseries(grid_cpu, b_timeseries; common_kwargs_geo_colorrange_b...,
                                                 Δ = Δn_snapshots)

    for i_snapshot in 1:(n_snapshots - 1)
        frame_index = floor(Int, i_snapshot * (n_frames - 1)/(n_snapshots - 1) + 1)
        simulation_time = (frame_index - 1) * save_fields_interval

        title = "Zonal velocity after $(prettytime(simulation_time))"
        set!(x_timeseries[frame_index], u_timeseries[frame_index])
        fig = geo_heatlatlon_visualization(grid_cpu, x_timeseries[frame_index], title; common_kwargs_geo...,
                                           cbar_label = "zonal velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = u_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_u_%d_geo_heatlatlon_plot.png", i_snapshot), fig)

        title = "Meridional velocity after $(prettytime(simulation_time))"
        set!(x_timeseries[frame_index], v_timeseries[frame_index])
        fig = geo_heatlatlon_visualization(grid_cpu, x_timeseries[frame_index], title; common_kwargs_geo...,
                                           cbar_label = "meridional velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = v_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_v_%d_geo_heatlatlon_plot.png", i_snapshot), fig)

        title = "Relative vorticity after $(prettytime(simulation_time))"
        set!(x_timeseries[frame_index], ζ_timeseries[frame_index])
        fig = geo_heatlatlon_visualization(grid_cpu, x_timeseries[frame_index], title; common_kwargs_geo...,
                                           cbar_label = "relative vorticity (s⁻¹)", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_ζ_%d_geo_heatlatlon_plot.png", i_snapshot), fig)

        title = "Surface elevation after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid_cpu, η_timeseries[frame_index], title; ssh = true,
                                           cbar_label = "surface elevation (m)", specify_plot_limits = true,
                                           plot_limits = η_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_η_%d_geo_heatlatlon_plot.png", i_snapshot), fig)

        title = "Buoyancy after $(prettytime(simulation_time))"
        fig = geo_heatlatlon_visualization(grid_cpu, b_timeseries[frame_index], title; common_kwargs_geo_b...,
                                           cbar_label = "buoyancy (m s⁻²)", specify_plot_limits = true,
                                           plot_limits = b_colorrange)
        save(@sprintf("cubed_sphere_aquaplanet_b_%d_geo_heatlatlon_plot.png", i_snapshot), fig)
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

    create_panel_wise_visualization_animation(grid_cpu, u_timeseries, framerate, "cubed_sphere_aquaplanet_u"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid_cpu, v_timeseries, framerate, "cubed_sphere_aquaplanet_v"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid_cpu, ζ_timeseries, framerate, "cubed_sphere_aquaplanet_ζ"; k = Nz,
                                              common_kwargs...)
    create_panel_wise_visualization_animation(grid_cpu, η_timeseries, framerate, "cubed_sphere_aquaplanet_η"; ssh = true)
    create_panel_wise_visualization_animation(grid_cpu, b_timeseries, framerate, "cubed_sphere_aquaplanet_b"; k = b_index,
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
            extract_field_at_specific_longitude_through_panel_center(grid_cpu, u_timeseries[i_frame], panel_index;
                                                                     levels = 1:Nz))

            v_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid_cpu, v_timeseries[i_frame], panel_index;
                                                                     levels = 1:Nz))

            ζ_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid_cpu, ζ_timeseries[i_frame], panel_index;
                                                                     levels = 1:Nz))

            η_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid_cpu, η_timeseries[i_frame], panel_index;
                                                                     levels = Nz+1:Nz+1))

            b_at_specific_longitude_through_panel_center[i_frame, :, :, index] = (
            extract_field_at_specific_longitude_through_panel_center(grid_cpu, b_timeseries[i_frame], panel_index;
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

    u_colorrange = specify_colorrange_timeseries(grid_cpu, u_timeseries; common_kwargs_geo_colorrange...)
    copy_to_center_field(x_timeseries, u_timeseries)
    geo_heatlatlon_visualization_animation(grid_cpu, x_timeseries, "cc", prettytimes, "Zonal velocity",
                                           "cubed_sphere_aquaplanet_u_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "zonal velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = u_colorrange, framerate = framerate)

    v_colorrange = specify_colorrange_timeseries(grid_cpu, v_timeseries; common_kwargs_geo_colorrange...)
    copy_to_center_field(x_timeseries, v_timeseries)
    geo_heatlatlon_visualization_animation(grid_cpu, x_timeseries, "cc", prettytimes, "Meridional velocity",
                                           "cubed_sphere_aquaplanet_v_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "meridional velocity (m s⁻¹)", specify_plot_limits = true,
                                           plot_limits = v_colorrange, framerate = framerate)

    ζ_colorrange = specify_colorrange_timeseries(grid_cpu, ζ_timeseries; common_kwargs_geo_colorrange...)
    copy_to_center_field(x_timeseries, ζ_timeseries)
    geo_heatlatlon_visualization_animation(grid_cpu, x_timeseries, "cc", prettytimes, "Relative vorticity",
                                           "cubed_sphere_aquaplanet_ζ_geo_heatlatlon_animation"; k = Nz,
                                           cbar_label = "relative vorticity (s⁻¹)", specify_plot_limits = true,
                                           plot_limits = ζ_colorrange, framerate = framerate)

    η_colorrange = specify_colorrange_timeseries(grid_cpu, η_timeseries; ssh = true)
    geo_heatlatlon_visualization_animation(grid_cpu, η_timeseries, "cc", prettytimes, "Surface elevation",
                                           "cubed_sphere_aquaplanet_η_geo_heatlatlon_animation"; ssh = true,
                                           cbar_label = "surface elevation (m)", specify_plot_limits = true,
                                           plot_limits = η_colorrange, framerate = framerate)

    b_colorrange = specify_colorrange_timeseries(grid_cpu, b_timeseries; common_kwargs_geo_colorrange_b...)
    geo_heatlatlon_visualization_animation(grid_cpu, b_timeseries, "cc", prettytimes, "Buoyancy",
                                           "cubed_sphere_aquaplanet_b_geo_heatlatlon_animation"; k = b_index,
                                           cbar_label = "buoyancy (m s⁻²)", specify_plot_limits = true,
                                           plot_limits = b_colorrange, framerate = framerate)
end
