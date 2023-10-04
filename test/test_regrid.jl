include("dependencies_for_runtests.jl")

using Oceananigans.Fields: regrid_in_x!, regrid_in_y!, regrid_in_z!

@testset "Field regridding" begin
    @info "  Testing field regridding..."
    
    L = 1.1
    ℓ = 0.5

    regular_ξ              = (0, L)
    fine_stretched_ξ       = [0, ℓ, L]
    very_fine_stretched_ξ  = [0, 0.2, 0.6, L]
    super_fine_stretched_ξ = [0, 0.1, 0.3, 0.65, L]
    
    topologies_1d = (x = (Bounded, Flat, Flat),
                     y = (Flat, Bounded, Flat),
                     z = (Flat, Flat, Bounded))

    sizes = (x = (2, 4, 6),
             y = (4, 2, 6),
             z = (4, 6, 2))

    topologies_3d = (x = (Bounded, Periodic, Periodic),
                     y = (Periodic, Bounded, Periodic),
                     z = (Periodic, Periodic, Bounded))

    regrid_xyz! = (x = regrid_in_x!,
                   y = regrid_in_y!,
                   z = regrid_in_z!)

    for arch in archs
        for dim in (:x, :y, :z)
            @testset "Regridding in $dim" begin
                regrid! = regrid_xyz![dim]
                topology = topologies_1d[dim]

                # 1D grids
                coarse_1d_regular_grid       = RectilinearGrid(arch, size=1; topology, Dict(dim => regular_ξ)...)
                fine_1d_regular_grid         = RectilinearGrid(arch, size=2; topology, Dict(dim => regular_ξ)...)
                fine_1d_stretched_grid       = RectilinearGrid(arch, size=2; topology, Dict(dim => fine_stretched_ξ)...)
                very_fine_1d_stretched_grid  = RectilinearGrid(arch, size=3; topology, Dict(dim => very_fine_stretched_ξ)...)
                super_fine_1d_stretched_grid = RectilinearGrid(arch, size=4; topology, Dict(dim => super_fine_stretched_ξ)...)
                super_fine_1d_regular_grid   = RectilinearGrid(arch, size=5; topology, Dict(dim => regular_ξ)...)

                # 3D grids
                topology = topologies_3d[dim]
                sz = sizes[dim]

                regular_kw = Dict{Any, Any}(d => (0, 1) for d in (:x, :y, :z) if d != dim)
                regular_kw[dim] = regular_ξ
                fine_regular_grid   = RectilinearGrid(arch, size=sz; topology, regular_kw...)

                fine_stretched_kw = Dict{Any, Any}(d => (0, 1) for d in (:x, :y, :z) if d != dim)
                fine_stretched_kw[dim] = fine_stretched_ξ
                fine_stretched_grid = RectilinearGrid(arch, size=sz; topology, fine_stretched_kw...)
               
                fine_stretched_c                    = CenterField(fine_stretched_grid)

                coarse_1d_regular_c                 = CenterField(coarse_1d_regular_grid)
                fine_1d_regular_c                   = CenterField(fine_1d_regular_grid)
                fine_1d_stretched_c                 = CenterField(fine_1d_stretched_grid)
                very_fine_1d_stretched_c            = CenterField(very_fine_1d_stretched_grid)
                super_fine_1d_stretched_c           = CenterField(super_fine_1d_stretched_grid)
                super_fine_1d_regular_c             = CenterField(super_fine_1d_regular_grid)
                super_fine_from_reduction_regular_c = CenterField(super_fine_1d_regular_grid)

                # We initialize an array on the `fine_1d_stretched_grid`, regrid it to the rest
                # grids, and check whether we get the anticipated results.
                c₁ = 1
                c₂ = 3

                CUDA.@allowscalar begin
                    interior(fine_1d_stretched_c)[1] = c₁
                    interior(fine_1d_stretched_c)[2] = c₂
                end

                # Coarse-graining
                regrid!(coarse_1d_regular_c, fine_1d_stretched_c)

                CUDA.@allowscalar begin
                    @test interior(coarse_1d_regular_c)[1] ≈ ℓ/L * c₁ + (1 - ℓ/L) * c₂
                end

                regrid!(fine_1d_regular_c, fine_1d_stretched_c)

                CUDA.@allowscalar begin
                    @test interior(fine_1d_regular_c)[1] ≈ ℓ/(L/2) * c₁ + (1 - ℓ/(L/2)) * c₂
                    @test interior(fine_1d_regular_c)[2] ≈ c₂
                end            

                # Fine-graining
                regrid!(very_fine_1d_stretched_c, fine_1d_stretched_c)

                CUDA.@allowscalar begin
                    @test interior(very_fine_1d_stretched_c)[1] ≈ c₁
                    @test interior(very_fine_1d_stretched_c)[2] ≈ (ℓ - 0.2)/0.4 * c₁ + (0.6 - ℓ)/0.4 * c₂
                    @test interior(very_fine_1d_stretched_c)[3] ≈ c₂
                end
                
                regrid!(super_fine_1d_stretched_c, fine_1d_stretched_c)

                CUDA.@allowscalar begin
                    @test interior(super_fine_1d_stretched_c)[1] ≈ c₁
                    @test interior(super_fine_1d_stretched_c)[2] ≈ c₁
                    @test interior(super_fine_1d_stretched_c)[3] ≈ (ℓ - 0.3)/0.35 * c₁ + (0.65 - ℓ)/0.35 * c₂
                    @test interior(super_fine_1d_stretched_c)[4] ≈ c₂
                end
                
                regrid!(super_fine_1d_regular_c, fine_1d_stretched_c)
                
                CUDA.@allowscalar begin
                    @test interior(super_fine_1d_regular_c)[1] ≈ c₁
                    @test interior(super_fine_1d_regular_c)[2] ≈ c₁
                    @test interior(super_fine_1d_regular_c)[3] ≈ (3 - ℓ/(L/5)) * c₂ + (-2 + ℓ/(L/5)) * c₁
                    @test interior(super_fine_1d_regular_c)[4] ≈ c₂
                    @test interior(super_fine_1d_regular_c)[5] ≈ c₂
                end

                #=
                # This test does not work, because we can only regrid in one direction.
                # To make this work, we have to transfer the reduced data to a "reduced" grid
                # (ie with one grid point in each reduced direction).
                
                # Fine-graining from reduction
                ind1 = dim == :x ? (1, :, :) : dim == :y ? (:, 1, :) : (:, :, 1)
                ind2 = dim == :x ? (2, :, :) : dim == :y ? (:, 2, :) : (:, :, 2)
                dims = dim == :x ? (2, 3) : dim == :y ? (1, 3) : (1, 2)

                Base.dotview(fine_stretched_c, ind1...) .= c₁
                Base.dotview(fine_stretched_c, ind2...) .= c₂
                
                fine_stretched_c_mean_xy = Field(Reduction(mean!, fine_stretched_c; dims))
                compute!(fine_stretched_c_mean_xy)

                @show size(fine_stretched_c_mean_xy.grid)
                @show size(super_fine_from_reduction_regular_c.grid)

                regrid!(super_fine_from_reduction_regular_c, fine_stretched_c_mean_xy)
                
                CUDA.@allowscalar begin
                    @test interior(super_fine_from_reduction_regular_c)[1] ≈ c₁
                    @test interior(super_fine_from_reduction_regular_c)[2] ≈ c₁
                    @test interior(super_fine_from_reduction_regular_c)[3] ≈ (3 - ℓ/(L/5)) * c₂ + (-2 + ℓ/(L/5)) * c₁
                    @test interior(super_fine_from_reduction_regular_c)[4] ≈ c₂
                    @test interior(super_fine_from_reduction_regular_c)[5] ≈ c₂
                end
                =#
            end
        end
    end
end
