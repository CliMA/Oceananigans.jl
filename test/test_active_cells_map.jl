include("dependencies_for_runtests.jl")

using Oceananigans.Operators: hack_cosd
using Oceananigans.ImmersedBoundaries: retrieve_surface_active_cells_map, 
                                       retrieve_interior_active_cells_map,
                                       immersed_cell

Nx = 16
Ny = 16
Nz = 10

@testset "Active cells map" begin
    for arch in archs
        underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, Nz),
                                                halo = (4, 4, 4),
                                                latitude = (-80, 80),
                                                longitude = (-160, 160),
                                                z = (-10, 0),
                                                radius = 1,
                                                topology=(Bounded, Bounded, Bounded))

        # Make sure the bottom is the same
        bottom_height = zeros(Nx, Ny)
        for i in 1:Nx, j in 1:Ny
            bottom_height[i, j] = - rand() * 10
        end

        bottom_height = on_architecture(arch, bottom_height)    
        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
        immersed_active_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

        @testset "Active cells map construction" begin
            surface_active_cells_map  = retrieve_surface_active_cells_map(immersed_active_grid)
            interior_active_cells_map = retrieve_interior_active_cells_map(immersed_active_grid, Val(:interior))

            surface_active_cells_map  = on_architecture(CPU(), surface_active_cells_map) 
            interior_active_cells_map = on_architecture(CPU(), interior_active_cells_map) 
            grid = on_architecture(CPU(), immersed_grid)

            for i in 1:Nx, j in 1:Ny, k in 1:Nz
                immersed = immersed_cell(i, j, k, grid)
                active = (i, j, k) ∈ interior_active_cells_map
                @test immersed ⊻ active
            end
            
            for i in 1:Nx, j in 1:Ny
                immersed = all(immersed_cell(i, j, k, grid) for k in 1:Nz)
                active = (i, j) ∈ surface_active_cells_map
                @test immersed ⊻ active
            end
        end

        @testset "Active cells map solid body rotation" begin

            ua, va, wa, ca, ηa = solid_body_rotation_test(immersed_active_grid)
            u, v, w, c, η      = solid_body_rotation_test(immersed_grid)

            ua = interior(on_architecture(CPU(), ua))
            va = interior(on_architecture(CPU(), va))
            wa = interior(on_architecture(CPU(), wa))
            ca = interior(on_architecture(CPU(), ca))
            ηa = interior(on_architecture(CPU(), ηa))

            u = interior(on_architecture(CPU(), u))
            v = interior(on_architecture(CPU(), v))
            w = interior(on_architecture(CPU(), w))
            c = interior(on_architecture(CPU(), c))
            η = interior(on_architecture(CPU(), η))

            atol = eps(eltype(immersed_grid))
            rtol = sqrt(eps(eltype(immersed_grid)))

            @test all(isapprox(u, ua; atol, rtol))
            @test all(isapprox(v, va; atol, rtol))
            @test all(isapprox(w, wa; atol, rtol))
            @test all(isapprox(c, ca; atol, rtol))
            @test all(isapprox(η, ηa; atol, rtol))
        end
    end
end