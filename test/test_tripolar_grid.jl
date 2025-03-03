include("dependencies_for_runtests.jl")

using Statistics: dot, norm
using Oceananigans.Utils: getregion
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.MultiRegion: ConformalCubedSphereGrid

@kernel function compute_nonorthogonality_angle!(angle, grid, xF, yF, zF)
    i, j = @index(Global, NTuple)
    
    @inbounds begin
        x⁻ = xF[i, j]
        y⁻ = yF[i, j]
        z⁻ = zF[i, j]

        x⁺¹ = xF[i + 1, j]
        y⁺¹ = yF[i + 1, j]
        z⁺¹ = zF[i + 1, j]
        x⁺² = xF[i, j + 1]
        y⁺² = yF[i, j + 1]
        z⁺² = zF[i, j + 1]
        
        v1 = (x⁺¹ - x⁻, y⁺¹ - y⁻, z⁺¹ - z⁻)
        v2 = (x⁺² - x⁻, y⁺² - y⁻, z⁺² - z⁻)

        # Check orthogonality by computing the angle between the vectors
        cosθ = dot(v1, v2) / (norm(v1) * norm(v2))
        immersed = immersed_cell(i, j, 1, grid)
        angle[i, j] = ifelse(immersed, π / 2, acos(cosθ)) - π / 2

        # convert to degrees
        angle[i, j] = rad2deg(angle[i, j])
    end
end

@testset "Orthogonality of family of ellipses and hyperbolae..." begin

    # Test the orthogonality of a tripolar grid based on the orthogonality of a 
    # cubed sphere of the same size (1ᵒ in latitude and longitude)
    cubed_sphere_grid = ConformalCubedSphereGrid(panel_size = (90, 90, 1), z = (0, 1))
    cubed_sphere_panel = getregion(cubed_sphere_grid, 1)

    angle_cubed_sphere = zeros(size(cubed_sphere_panel)...)
    cartesian_nodes, _ = get_cartesian_nodes_and_vertices(cubed_sphere_panel, Face(), Face(), Center())
    xF, yF, zF = cartesian_nodes
    Nx, Ny, _  = size(cubed_sphere_panel)

    # Exclude the corners from the computation! (They are definitely not orthogonal)
    params = KernelParameters(5:Nx-5, 5:Ny-5)

    launch!(CPU(), cubed_sphere_panel, params, compute_nonorthogonality_angle!, angle_cubed_sphere, cubed_sphere_panel, xF, yF, zF)

    first_pole_longitude = λ¹ₚ = 75
    north_poles_latitude = φₚ  = 35
    
    λ²ₚ = λ¹ₚ + 180

    # Build a tripolar grid at 1ᵒ
    underlying_grid = TripolarGrid(; size = (360, 180, 1), first_pole_longitude, north_poles_latitude)

    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λ¹ₚ) < 5) & (abs(φₚ - φ) < 5)) |
                          ((abs(λ - λ²ₚ) < 5) & (abs(φₚ - φ) < 5)) | (φ < -78) ? 1 : 0

    # Exclude the singularities from the computation! (They are definitely not orthogonal)
    tripolar_grid      = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
    angle_tripolar     = zeros(size(tripolar_grid)...)
    cartesian_nodes, _ = get_cartesian_nodes_and_vertices(tripolar_grid.underlying_grid, Face(), Face(), Center())
    xF, yF, zF = cartesian_nodes
    Nx, Ny, _  = size(tripolar_grid)

    launch!(CPU(), tripolar_grid, (Nx-1, Ny-1), compute_nonorthogonality_angle!, angle_tripolar, tripolar_grid, xF, yF, zF)

    @test maximum(angle_tripolar) < maximum(angle_cubed_sphere)
    @test minimum(angle_tripolar) > minimum(angle_cubed_sphere)
end