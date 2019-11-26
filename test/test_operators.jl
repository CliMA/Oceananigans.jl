function test_function_differentiation(T=Float64)
    grid = RegularCartesianGrid(T; size=(3, 3, 3), length=(3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ∂x_ϕ_f = ϕ²[2, 2, 2] - ϕ²[1, 2, 2]
    ∂x_ϕ_c = ϕ²[3, 2, 2] - ϕ²[2, 2, 2]

    ∂y_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 1, 2]
    ∂y_ϕ_c = ϕ²[2, 3, 2] - ϕ²[2, 2, 2]

    ∂z_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 2, 1]
    ∂z_ϕ_c = ϕ²[2, 2, 3] - ϕ²[2, 2, 2]

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        ∂xᶜᵃᵃ(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_c &&
        ∂xᶠᵃᵃ(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_f &&

        ∂yᵃᶜᵃ(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_c &&
        ∂yᵃᶠᵃ(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_f &&

        ∂zᵃᵃᶜ(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_c &&
        ∂zᵃᵃᶠ(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_f
    )
end

function test_function_interpolation(T=Float64)
    grid = RegularCartesianGrid(T; size=(3, 3, 3), length=(3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ℑx_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[1, 2, 2]) / 2
    ℑx_ϕ_c = (ϕ²[3, 2, 2] + ϕ²[2, 2, 2]) / 2

    ℑy_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 1, 2]) / 2
    ℑy_ϕ_c = (ϕ²[2, 3, 2] + ϕ²[2, 2, 2]) / 2

    ℑz_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 2, 1]) / 2
    ℑz_ϕ_c = (ϕ²[2, 2, 3] + ϕ²[2, 2, 2]) / 2

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        ℑxᶜᵃᵃ(2, 2, 2, grid, f, ϕ) == ℑx_ϕ_c &&
        ℑxᶠᵃᵃ(2, 2, 2, grid, f, ϕ) == ℑx_ϕ_f &&

        ℑyᵃᶜᵃ(2, 2, 2, grid, f, ϕ) == ℑy_ϕ_c &&
        ℑyᵃᶠᵃ(2, 2, 2, grid, f, ϕ) == ℑy_ϕ_f &&

        ℑzᵃᵃᶜ(2, 2, 2, grid, f, ϕ) == ℑz_ϕ_c &&
        ℑzᵃᵃᶠ(2, 2, 2, grid, f, ϕ) == ℑz_ϕ_f
    )
end

@testset "Operators" begin
    println("Testing operators...")

    @testset "Function differentiation" begin
        println("  Testing function differentiation...")
        @test test_function_differentiation()
    end

    @testset "Function interpolation" begin
        println("  Testing function interpolation...")
        @test test_function_interpolation()
    end

    @testset "2D operators" begin
        println("  Testing 2D operators...")

        Nx, Ny, Nz = 32, 16, 8
        Lx, Ly, Lz = 100, 100, 100

        arch = CPU()
        grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))

        Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
        Tx, Ty, Tz = grid.Tx, grid.Ty, grid.Tz

        A3 = OffsetArray(zeros(Tx, Ty, Tz), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1-Hz:Nz+Hz)
        @. @views A3[1:Nx, 1:Ny, 1:Nz] = rand()
        fill_halo_regions!(A3, HorizontallyPeriodicBCs(), arch, grid)

        # A yz-slice with Nx==1.
        A2yz = OffsetArray(zeros(1+2Hx, Ty, Tz), 1-Hx:1+Hx, 1-Hy:Ny+Hy, 1-Hz:Nz+Hz)
        grid_yz = RegularCartesianGrid(size=(1, Ny, Nz), length=(Lx, Ly, Lz))

        # Manually fill in halos for the slice.
        A2yz[0:2, 0:Ny+1, 1:Nz] .= A3[1:1, 0:Ny+1, 1:Nz]
        A2yz[:, :, 0] .= A2yz[:, :, 1]
        A2yz[:, :, Nz+1] .= A2yz[:, :, Nz]

        # An xz-slice with Ny==1.
        A2xz = OffsetArray(zeros(Tx, 1+2Hy, Tz), 1-Hx:Nx+Hx, 1-Hy:1+Hy, 1-Hz:Nz+Hz)
        grid_xz = RegularCartesianGrid(size=(Nx, 1, Nz), length=(Lx, Ly, Lz))

        # Manually fill in halos for the slice.
        A2xz[0:Nx+1, 0:2, 1:Nz] .= A3[0:Nx+1, 1:1, 1:Nz]
        A2xz[:, :, 0] .= A2xz[:, :, 1]
        A2xz[:, :, Nz+1] .= A2xz[:, :, Nz]

        test_indices_3d = [(4, 5, 5), (21, 11, 4), (16, 8, 4),  (30, 12, 3), (11, 3, 6), # Interior
                           (2, 10, 4), (31, 5, 6), (10, 2, 4), (17, 15, 5), (17, 10, 2), (23, 5, 7),  # Borderlands
                           (1, 5, 5), (32, 10, 3), (16, 1, 4), (16, 16, 4), (16, 8, 1), (16, 8, 8),  # Edges
                           (1, 1, 1), (32, 16, 8)] # Corners

        test_indices_2d_yz = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
                              (1, 1, 5), (1, 5, 1), (1, 5, 5), (1, 11, 4),
                              (1, 15, 7), (1, 15, 8), (1, 16, 7), (1, 16, 8)]

        test_indices_2d_xz = [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2),
                              (1, 1, 5), (5, 1, 1), (5, 1, 5), (17, 1, 4),
                              (31, 1, 7), (31, 1, 8), (32, 1, 7), (32, 1, 8)]

        for idx in test_indices_2d_yz
            @test δxᶜᵃᵃ(idx..., grid_yz, A2yz) ≈ 0
            @test δxᶠᵃᵃ(idx..., grid_yz, A2yz) ≈ 0
            @test δyᵃᶜᵃ(idx..., grid_yz, A2yz) ≈ δyᵃᶜᵃ(idx..., grid_yz, A3)
            @test δyᵃᶠᵃ(idx..., grid_yz, A2yz) ≈ δyᵃᶠᵃ(idx..., grid_yz, A3)
            @test δzᵃᵃᶜ(idx..., grid_yz, A2yz) ≈ δzᵃᵃᶜ(idx..., grid_yz, A3)
            @test δzᵃᵃᶠ(idx..., grid_yz, A2yz) ≈ δzᵃᵃᶠ(idx..., grid_yz, A3)
        end

        for idx in test_indices_2d_xz
            @test δxᶜᵃᵃ(idx..., grid_xz, A2xz) ≈ δxᶜᵃᵃ(idx..., grid_xz, A3)
            @test δxᶠᵃᵃ(idx..., grid_xz, A2xz) ≈ δxᶠᵃᵃ(idx..., grid_xz, A3)
            @test δyᵃᶜᵃ(idx..., grid_xz, A2xz) ≈ 0
            @test δyᵃᶠᵃ(idx..., grid_xz, A2xz) ≈ 0
            @test δzᵃᵃᶜ(idx..., grid_xz, A2xz) ≈ δzᵃᵃᶜ(idx..., grid_xz, A3)
            @test δzᵃᵃᶠ(idx..., grid_xz, A2xz) ≈ δzᵃᵃᶠ(idx..., grid_xz, A3)
        end
    end
end
