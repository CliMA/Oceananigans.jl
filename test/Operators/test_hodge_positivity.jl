using Test
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Azᶠᶜᶜ, Azᶜᶠᶜ
using Oceananigans.Operators: covariant_to_volume_flux_uᶠᶜᶜ, covariant_to_volume_flux_vᶜᶠᶜ
using LinearAlgebra

function octahealpix_independent_hodge_matrix(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              mapping = OctaHEALPixMapping(N),
                              size = (2N, 2N, 1),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              halo = (5, 5, 3))

    Nx, Ny, _ = size(grid)
    Nu = Nx * Ny
    Nv = Nx * Ny
    degrees_of_freedom = Nu + Nv

    u_index(i, j) = (j - 1) * Nx + i
    v_index(i, j) = Nu + (j - 1) * Nx + i

    u = XFaceField(grid)
    v = YFaceField(grid)
    coefficients = zeros(FT, degrees_of_freedom)
    hodge_matrix = zeros(FT, degrees_of_freedom, degrees_of_freedom)

    function set_independent_face_values!(coefficients)
        fill!(parent(u), zero(FT))
        fill!(parent(v), zero(FT))

        for j in 1:Ny, i in 1:Nx
            u[i, j, 1] = coefficients[u_index(i, j)]
            v[i, j, 1] = coefficients[v_index(i, j)]
        end

        fill_halo_regions!((u, v))

        return nothing
    end

    function independent_flux_vector!()
        fluxes = zeros(FT, degrees_of_freedom)

        for j in 1:Ny, i in 1:Nx
            fluxes[u_index(i, j)] = covariant_to_volume_flux_uᶠᶜᶜ(i, j, 1, grid, u, v)
            fluxes[v_index(i, j)] = covariant_to_volume_flux_vᶜᶠᶜ(i, j, 1, grid, u, v)
        end

        return fluxes
    end

    for column in 1:degrees_of_freedom
        fill!(coefficients, zero(FT))
        coefficients[column] = one(FT)
        set_independent_face_values!(coefficients)
        hodge_matrix[:, column] .= independent_flux_vector!()
    end

    weights = zeros(FT, degrees_of_freedom)

    for j in 1:Ny, i in 1:Nx
        weights[u_index(i, j)] = Azᶠᶜᶜ(i, j, 1, grid) / 2
        weights[v_index(i, j)] = Azᶜᶠᶜ(i, j, 1, grid) / 2
    end

    return hodge_matrix, weights
end

function octahealpix_independent_hodge_diagnostics(FT, N)
    hodge_matrix, weights = octahealpix_independent_hodge_matrix(FT, N)
    weighted_hodge = Diagonal(weights) * hodge_matrix
    symmetric_part = (weighted_hodge + weighted_hodge') / 2
    skew_part = (weighted_hodge - weighted_hodge') / 2
    eigenvalues = eigvals(Symmetric(symmetric_part))
    adjointness_defect = opnorm(skew_part) / max(opnorm(symmetric_part), eps(FT))

    return (; adjointness_defect,
              minimum_eigenvalue = minimum(eigenvalues),
              maximum_eigenvalue = maximum(eigenvalues),
              negative_eigenvalues = count(<(-sqrt(eps(FT))), eigenvalues))
end

@testset "OctaHEALPix independent Hodge positivity and adjointness" begin
    for FT in (Float32, Float64)
        diagnostics = octahealpix_independent_hodge_diagnostics(FT, 4)
        tolerance = sqrt(eps(FT))

        @info "OctaHEALPix independent Hodge diagnostics" FT diagnostics.adjointness_defect diagnostics.minimum_eigenvalue diagnostics.maximum_eigenvalue diagnostics.negative_eigenvalues

        @test diagnostics.adjointness_defect ≤ tolerance
        @test diagnostics.minimum_eigenvalue > zero(FT)
        @test diagnostics.maximum_eigenvalue > diagnostics.minimum_eigenvalue
        @test diagnostics.negative_eigenvalues == 0
    end
end
