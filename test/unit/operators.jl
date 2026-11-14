include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Operators: Δxᶠᵃᵃ, Δxᶜᵃᵃ, Δxᶠᶠᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ
using Oceananigans.Operators: Δyᵃᶠᵃ, Δyᵃᶜᵃ, Δyᶠᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ
using Oceananigans.BoundaryConditions: UPivot, TPivot, FPivot, regularize_field_boundary_conditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Operators: ℑxTᶠᵃᵃ, ℑxTᶜᵃᵃ, ℑyTᵃᶠᵃ, ℑyTᵃᶜᵃ

# Fill a field's interior with random data and its halos with `NaN`, so that any operator
# reading a halo (rather than wrapping or clamping its index) fails an `≈` comparison.
function randomize_interior_and_poison_halos!(field)
    set!(field, rand(size(field)...))
    data = Array(interior(field))
    parent(field) .= NaN
    interior(field) .= data
    return field
end

# A function argument that distinguishes the grid it is evaluated on: topology-aware
# operators on an `ImmersedBoundaryGrid` must pass the immersed grid itself to `f`.
grid_marked_value(i, j, k, grid::ImmersedBoundaryGrid, c) = @inbounds 10 * c[i, j, k]
grid_marked_value(i, j, k, grid, c) = @inbounds c[i, j, k]


function test_three_dimensional_differences(T=Float64)
    grid = RectilinearGrid(CPU(), T; size=(3, 3, 3), extent=(3, 3, 3))
    ϕ = rand(T, 3, 3, 3)

    grid = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> x < 1))

    ϕ²  = ϕ.^2

    δx_ϕ_f = T(0)
    δx_ϕ_c = ϕ²[3, 2, 2] - ϕ²[2, 2, 2]

    δy_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 1, 2]
    δy_ϕ_c = ϕ²[2, 3, 2] - ϕ²[2, 2, 2]

    δz_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 2, 1]
    δz_ϕ_c = ϕ²[2, 2, 3] - ϕ²[2, 2, 2]

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    for δx in (δxᶜᶜᶜ, δxᶜᶜᶠ, δxᶜᶠᶜ, δxᶜᶠᶠ)
        @test δx(2, 2, 2, grid, f, ϕ) == δx_ϕ_c
    end
    for δx in (∂xᶠᶜᶜ, δxᶠᶜᶠ, δxᶠᶠᶜ, δxᶠᶠᶠ)
        @test δx(2, 2, 2, grid, f, ϕ) == δx_ϕ_f
    end

    for δy in (∂yᶜᶜᶜ, δyᶜᶜᶠ, δyᶠᶜᶜ, δyᶠᶜᶠ)
        @test δy(2, 2, 2, grid, f, ϕ) == δy_ϕ_c
    end
    for δy in (δyᶜᶠᶜ, δyᶠᶠᶜ, δyᶜᶠᶠ, δyᶠᶠᶠ)
        @test δy(2, 2, 2, grid, f, ϕ) == δy_ϕ_f
    end

    for δz in (δzᶜᶜᶜ, δzᶜᶠᶜ, δzᶠᶜᶜ, δzᶠᶠᶜ)
        @test δz(2, 2, 2, grid, f, ϕ) == δz_ϕ_c
    end
    for δz in (δzᶜᶜᶠ, δzᶜᶠᶠ, δzᶠᶜᶠ, δzᶠᶠᶠ)
        @test δz(2, 2, 2, grid, f, ϕ) == δz_ϕ_f
    end

    # the x-face between the active cell 2 and the immersed cell 3 is active
    mirrored_grid = ImmersedBoundaryGrid(grid.underlying_grid, GridFittedBoundary((x, y, z) -> x > 2))
    @test δyᶠᶠᶜ(3, 2, 2, mirrored_grid, f, ϕ) == ϕ²[3, 2, 2] - ϕ²[3, 1, 2]

    return nothing
end

function test_function_differentiation(T=Float64)
    grid = RectilinearGrid(CPU(), T; size=(3, 3, 3), extent=(3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ∂x_ϕ_f = ϕ²[2, 2, 2] - ϕ²[1, 2, 2]
    ∂x_ϕ_c = ϕ²[3, 2, 2] - ϕ²[2, 2, 2]

    ∂y_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 1, 2]
    ∂y_ϕ_c = ϕ²[2, 3, 2] - ϕ²[2, 2, 2]

    ∂z_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 2, 1]
    ∂z_ϕ_c = ϕ²[2, 2, 3] - ϕ²[2, 2, 2]

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    for ∂x in (∂xᶜᶜᶜ, ∂xᶜᶜᶠ, ∂xᶜᶠᶜ, ∂xᶜᶠᶠ)
        @test ∂x(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_c
    end
    for ∂x in (∂xᶠᶜᶜ, ∂xᶠᶜᶠ, ∂xᶠᶠᶜ, ∂xᶠᶠᶠ)
        @test ∂x(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_f
    end

    for ∂y in (∂yᶜᶜᶜ, ∂yᶜᶜᶠ, ∂yᶠᶜᶜ, ∂yᶠᶜᶠ)
        @test ∂y(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_c
    end
    for ∂y in (∂yᶜᶠᶜ, ∂yᶠᶠᶜ, ∂yᶜᶠᶠ, ∂yᶠᶠᶠ)
        @test ∂y(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_f
    end

    for ∂z in (∂zᶜᶜᶜ, ∂zᶜᶠᶜ, ∂zᶠᶜᶜ, ∂zᶠᶠᶜ)
        @test ∂z(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_c
    end
    for ∂z in (∂zᶜᶜᶠ, ∂zᶜᶠᶠ, ∂zᶠᶜᶠ, ∂zᶠᶠᶠ)
        @test ∂z(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_f
    end

    stretched_f = [0, 1, 3, 6]
    stretched_c = StaticStridesOffsetArray([-0.5, 0.5, 2, 4.5, 7.5], -1)

    dc(i) = stretched_f[i+1] - stretched_f[i]
    df(i) = stretched_c[i] - stretched_c[i-1]

    grid  = RectilinearGrid(CPU(), T; size=(3, 3, 3), x=stretched_f, y=stretched_f, z=stretched_f, topology = (Bounded, Bounded, Bounded))

    ∂x_f(i, j, k) = 1 / df(i) * (ϕ²[i, j, k]   - ϕ²[i-1, j, k])
    ∂x_c(i, j, k) = 1 / dc(i) * (ϕ²[i+1, j, k] - ϕ²[i, j, k])
    ∂y_f(i, j, k) = 1 / df(j) * (ϕ²[i, j, k]   - ϕ²[i, j-1, k])
    ∂y_c(i, j, k) = 1 / dc(j) * (ϕ²[i, j+1, k] - ϕ²[i, j, k])
    ∂z_f(i, j, k) = 1 / df(k) * (ϕ²[i, j, k]   - ϕ²[i, j, k-1])
    ∂z_c(i, j, k) = 1 / dc(k) * (ϕ²[i, j, k+1] - ϕ²[i, j, k])

    for ∂x in (∂xᶜᶜᶜ, ∂xᶜᶜᶠ, ∂xᶜᶠᶜ, ∂xᶜᶠᶠ)
        @test ∂x(2, 2, 2, grid, f, ϕ) == ∂x_c(2, 2, 2)
    end
    for ∂x in (∂xᶠᶜᶜ, ∂xᶠᶜᶠ, ∂xᶠᶠᶜ, ∂xᶠᶠᶠ)
        @test ∂x(2, 2, 2, grid, f, ϕ) == ∂x_f(2, 2, 2)
    end

    for ∂y in (∂yᶜᶜᶜ, ∂yᶜᶜᶠ, ∂yᶠᶜᶜ, ∂yᶠᶜᶠ)
        @test ∂y(2, 2, 2, grid, f, ϕ) == ∂y_c(2, 2, 2)
    end
    for ∂y in (∂yᶜᶠᶜ, ∂yᶠᶠᶜ, ∂yᶜᶠᶠ, ∂yᶠᶠᶠ)
        @test ∂y(2, 2, 2, grid, f, ϕ) == ∂y_f(2, 2, 2)
    end

    for ∂z in (∂zᶜᶜᶜ, ∂zᶜᶠᶜ, ∂zᶠᶜᶜ, ∂zᶠᶠᶜ)
        @test ∂z(2, 2, 2, grid, f, ϕ) == ∂z_c(2, 2, 2)
    end
    for ∂z in (∂zᶜᶜᶠ, ∂zᶜᶠᶠ, ∂zᶠᶜᶠ, ∂zᶠᶠᶠ)
        @test ∂z(2, 2, 2, grid, f, ϕ) == ∂z_f(2, 2, 2)
    end

    return nothing
end

function test_function_interpolation(T=Float64)
    grid = RectilinearGrid(CPU(), T; size=(3, 3, 3), extent=(3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ℑx_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[1, 2, 2]) / 2
    ℑx_ϕ_c = (ϕ²[3, 2, 2] + ϕ²[2, 2, 2]) / 2

    ℑy_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 1, 2]) / 2
    ℑy_ϕ_c = (ϕ²[2, 3, 2] + ϕ²[2, 2, 2]) / 2

    ℑz_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 2, 1]) / 2
    ℑz_ϕ_c = (ϕ²[2, 2, 3] + ϕ²[2, 2, 2]) / 2

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    @test ℑxᶜᵃᵃ(2, 2, 2, grid, f, ϕ) == ℑx_ϕ_c
    @test ℑxᶠᵃᵃ(2, 2, 2, grid, f, ϕ) == ℑx_ϕ_f

    @test ℑyᵃᶜᵃ(2, 2, 2, grid, f, ϕ) == ℑy_ϕ_c
    @test ℑyᵃᶠᵃ(2, 2, 2, grid, f, ϕ) == ℑy_ϕ_f

    @test ℑzᵃᵃᶜ(2, 2, 2, grid, f, ϕ) == ℑz_ϕ_c
    @test ℑzᵃᵃᶠ(2, 2, 2, grid, f, ϕ) == ℑz_ϕ_f

    return nothing
end

@testset "Operators" begin
    @info "Testing operators..."

    @testset "Grid lengths, areas, and volume operators" begin
        @info "  Testing grid lengths, areas, and volume operators..."

        x_spacings = ( [eval(Symbol(:Δx, LX, :ᵃ, :ᵃ)) for LX in (:ᶜ, :ᶠ)]...,
                       [eval(Symbol(:Δx, LX, LY, :ᵃ)) for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)]...,
                       [eval(Symbol(:Δx, LX, LY, LZ)) for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)]...)

        y_spacings = ( [eval(Symbol(:Δy, :ᵃ, LY, :ᵃ)) for LY in (:ᶜ, :ᶠ)]...,
                       [eval(Symbol(:Δy, LX, LY, :ᵃ)) for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)]...,
                       [eval(Symbol(:Δy, LX, LY, LZ)) for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)]...)

        z_spacings = ( [eval(Symbol(:Δz, :ᵃ, :ᵃ, LZ)) for LZ in (:ᶜ, :ᶠ)]...,
                       [eval(Symbol(:Δz, LX, LY, LZ)) for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)]...)

        FT = Float64
        grid = RectilinearGrid(CPU(), FT, size=(1, 1, 1), extent=(π, 2π, 3π))

        @testset "Easterly lengths" begin
            @info "    Testing easterly lengths..."
            for δ in x_spacings
                @test δ(1, 1, 1, grid) == FT(π)
            end
        end

        @testset "Westerly lengths" begin
            @info "    Testing westerly lengths..."
            for δ in y_spacings
                @test δ(1, 1, 1, grid) == FT(2π)
            end
        end

        @testset "Vertical lengths" begin
            @info "    Testing vertical lengths..."
            for δ in z_spacings
                @test δ(1, 1, 1, grid) == FT(3π)
            end
        end

        @testset "East-normal areas in the yz-plane" begin
            @info "    Testing areas with easterly normal in the yz-plane..."
            for A in (Axᶜᶜᶜ, Axᶠᶜᶜ, Axᶜᶠᶜ, Axᶜᶜᶠ, Axᶠᶠᶠ, Axᶠᶠᶜ, Axᶠᶜᶠ, Axᶜᶠᶠ)
                @test A(1, 1, 1, grid) == FT(6 * π^2)
            end
        end

        @testset "West-normal areas in the xz-plane" begin
            @info "    Testing areas with westerly normal in the xz-plane..."
            for A in (Ayᶜᶜᶜ, Ayᶠᶜᶜ, Ayᶜᶠᶜ, Ayᶜᶜᶠ, Ayᶠᶠᶠ, Ayᶠᶠᶜ, Ayᶠᶜᶠ, Ayᶜᶠᶠ)
                @test A(1, 1, 1, grid) == FT(3 * π^2)
            end
        end

        @testset "Horizontal areas in the xy-plane" begin
            @info "    Testing horizontal areas in the xy-plane..."
            for A in (Azᶜᶜᶜ, Azᶠᶜᶜ, Azᶜᶠᶜ, Azᶜᶜᶠ, Azᶠᶠᶠ, Azᶠᶠᶜ, Azᶠᶜᶠ, Azᶜᶠᶠ)
                @test A(1, 1, 1, grid) == FT(2 * π^2)
            end
        end

        @testset "Volumes" begin
            @info "    Testing volumes..."
            for V in (Vᶜᶜᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ, Vᶠᶠᶠ, Vᶠᶠᶜ, Vᶠᶜᶠ, Vᶜᶠᶠ)
                @test V(1, 1, 1, grid) == FT(6 * π^3)
            end
        end

    end

    @testset "Function differences" begin
        @info "  Testing function differences..."
        test_three_dimensional_differences()
    end

    @testset "Function differentiation" begin
        @info "  Testing function differentiation..."
        test_function_differentiation()
    end

    @testset "Function interpolation" begin
        @info "  Testing function interpolation..."
        test_function_interpolation()
    end

    @testset "2D operators" begin
        @info "  Testing 2D operators..."

        Nx, Ny, Nz = 32, 16, 8
        Lx, Ly, Lz = 100, 100, 100

        arch = CPU()
        grid = RectilinearGrid(CPU(), size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()))

        Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
        Tx, Ty, Tz = Nx+2Hx, Ny+2Hy, Nz+2Hz

        A3 = StaticStridesOffsetArray(zeros(Tx, Ty, Tz), -Hx, -Hy, -Hz)
        @. @views A3[1:Nx, 1:Ny, 1:Nz] = rand()
        fill_halo_regions!(A3, bcs, (:, :, :), (Center(), Center(), Center()), grid)

        # A yz-slice with Nx==1.
        A2yz = StaticStridesOffsetArray(zeros(1+2Hx, Ty, Tz), -Hx, -Hy, -Hz)
        grid_yz = RectilinearGrid(CPU(), size=(1, Ny, Nz), extent=(Lx, Ly, Lz))

        # Manually fill in halos for the slice.
        A2yz[0:2, 0:Ny+1, 1:Nz] .= A3[1:1, 0:Ny+1, 1:Nz]
        A2yz[:, :, 0] .= A2yz[:, :, 1]
        A2yz[:, :, Nz+1] .= A2yz[:, :, Nz]

        # An xz-slice with Ny==1.
        A2xz = StaticStridesOffsetArray(zeros(Tx, 1+2Hy, Tz), -Hx, -Hy, -Hz)
        grid_xz = RectilinearGrid(CPU(), size=(Nx, 1, Nz), extent=(Lx, Ly, Lz))

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

    @testset "Topology-aware operators on tripolar grids" begin
        for (fold_topology, pivot) in ((CenterFolded, UPivot), (CenterFolded, TPivot), (FaceFolded, FPivot))
            grid = TripolarGrid(CPU(); size = (16, 10, 1), z = (-1, 0), fold_topology, pivot)
            Nx, Ny, _ = size(grid)
            northernmost_row = fold_topology === FaceFolded ? Ny + 1 : Ny

            U = Field{Face, Center, Nothing}(grid; boundary_conditions = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :U))
            V = Field{Center, Face, Nothing}(grid; boundary_conditions = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :V))
            c = Field{Center, Center, Nothing}(grid)

            for f in (U, V, c)
                set!(f, rand(size(f)...))
                fill_halo_regions!(f)
            end

            # A topology-aware operator equals the plain operator applied to a halo-filled field
            @test all(δxTᶜᵃᵃ(i, j, 1, grid, U) ≈ δxᶜᵃᵃ(i, j, 1, grid, U) for i in 1:Nx, j in 1:Ny)
            @test all(δxTᶠᵃᵃ(i, j, 1, grid, c) ≈ δxᶠᵃᵃ(i, j, 1, grid, c) for i in 1:Nx, j in 1:Ny)
            @test all(δyTᵃᶜᵃ(i, j, 1, grid, V) ≈ δyᵃᶜᵃ(i, j, 1, grid, V) for i in 1:Nx, j in 1:northernmost_row)
            @test all(δyTᵃᶠᵃ(i, j, 1, grid, c) ≈ δyᵃᶠᵃ(i, j, 1, grid, c) for i in 1:Nx, j in 1:northernmost_row)

            @test all(ℑxTᶜᵃᵃ(i, j, 1, grid, U) ≈ ℑxᶜᵃᵃ(i, j, 1, grid, U) for i in 1:Nx, j in 1:Ny)
            @test all(ℑxTᶠᵃᵃ(i, j, 1, grid, c) ≈ ℑxᶠᵃᵃ(i, j, 1, grid, c) for i in 1:Nx, j in 1:Ny)
            @test all(ℑyTᵃᶜᵃ(i, j, 1, grid, V) ≈ ℑyᵃᶜᵃ(i, j, 1, grid, V) for i in 1:Nx, j in 1:northernmost_row)
            @test all(ℑyTᵃᶠᵃ(i, j, 1, grid, c) ≈ ℑyᵃᶠᵃ(i, j, 1, grid, c) for i in 1:Nx, j in 1:northernmost_row)
        end
    end

    @testset "Topology-aware operators on immersed boundary grids" begin
        underlying = RectilinearGrid(CPU(), size=(8, 6, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                                     topology=(Periodic, Periodic, Bounded))
        grid = ImmersedBoundaryGrid(underlying, GridFittedBottom((x, y) -> -0.5 - 0.3 * sinpi(2x) * cospi(2y)))
        Nx, Ny, Nz = size(grid)

        c = randomize_interior_and_poison_halos!(CenterField(grid))
        U = randomize_interior_and_poison_halos!(XFaceField(grid))
        V = randomize_interior_and_poison_halos!(YFaceField(grid))

        interior_indices = Iterators.product(1:Nx, 1:Ny, 1:Nz)

        # The immersed grid dispatches to the periodic methods, which wrap around at the edges
        # rather than reading the (poisoned) halos, and agree with the underlying grid.
        @test all(δxTᶠᵃᵃ(i, j, k, grid, c) ≈ c[i, j, k] - c[mod1(i - 1, Nx), j, k] for (i, j, k) in interior_indices)
        @test all(δyTᵃᶠᵃ(i, j, k, grid, c) ≈ c[i, j, k] - c[i, mod1(j - 1, Ny), k] for (i, j, k) in interior_indices)
        @test all(δxTᶜᵃᵃ(i, j, k, grid, U) ≈ U[mod1(i + 1, Nx), j, k] - U[i, j, k] for (i, j, k) in interior_indices)
        @test all(δyTᵃᶜᵃ(i, j, k, grid, V) ≈ V[i, mod1(j + 1, Ny), k] - V[i, j, k] for (i, j, k) in interior_indices)

        @test all(δxTᶠᵃᵃ(i, j, k, grid, c) == δxTᶠᵃᵃ(i, j, k, underlying, c) for (i, j, k) in interior_indices)
        @test all(δyTᵃᶠᵃ(i, j, k, grid, c) == δyTᵃᶠᵃ(i, j, k, underlying, c) for (i, j, k) in interior_indices)
        @test all(δxTᶜᵃᵃ(i, j, k, grid, U) == δxTᶜᵃᵃ(i, j, k, underlying, U) for (i, j, k) in interior_indices)
        @test all(δyTᵃᶜᵃ(i, j, k, grid, V) == δyTᵃᶜᵃ(i, j, k, underlying, V) for (i, j, k) in interior_indices)

        @test all(ℑxTᶠᵃᵃ(i, j, k, grid, c) ≈ (c[i, j, k] + c[mod1(i - 1, Nx), j, k]) / 2 for (i, j, k) in interior_indices)
        @test all(ℑyTᵃᶠᵃ(i, j, k, grid, c) ≈ (c[i, j, k] + c[i, mod1(j - 1, Ny), k]) / 2 for (i, j, k) in interior_indices)
        @test all(ℑxTᶜᵃᵃ(i, j, k, grid, U) ≈ (U[mod1(i + 1, Nx), j, k] + U[i, j, k]) / 2 for (i, j, k) in interior_indices)
        @test all(ℑyTᵃᶜᵃ(i, j, k, grid, V) ≈ (V[i, mod1(j + 1, Ny), k] + V[i, j, k]) / 2 for (i, j, k) in interior_indices)

        # A function argument is evaluated on the immersed grid itself, not on its underlying grid
        for (i, j, k) in ((1, 1, 1), (Nx, Ny, Nz), (3, 4, 2))
            @test δxTᶠᵃᵃ(i, j, k, grid, grid_marked_value, c) ≈ 10 * δxTᶠᵃᵃ(i, j, k, grid, c)
            @test δyTᵃᶠᵃ(i, j, k, grid, grid_marked_value, c) ≈ 10 * δyTᵃᶠᵃ(i, j, k, grid, c)
            @test δxTᶜᵃᵃ(i, j, k, grid, grid_marked_value, U) ≈ 10 * δxTᶜᵃᵃ(i, j, k, grid, U)
            @test δyTᵃᶜᵃ(i, j, k, grid, grid_marked_value, V) ≈ 10 * δyTᵃᶜᵃ(i, j, k, grid, V)
            @test ℑxTᶠᵃᵃ(i, j, k, grid, grid_marked_value, c) ≈ 10 * ℑxTᶠᵃᵃ(i, j, k, grid, c)
            @test ℑyTᵃᶠᵃ(i, j, k, grid, grid_marked_value, c) ≈ 10 * ℑyTᵃᶠᵃ(i, j, k, grid, c)
            @test ℑxTᶜᵃᵃ(i, j, k, grid, grid_marked_value, U) ≈ 10 * ℑxTᶜᵃᵃ(i, j, k, grid, U)
            @test ℑyTᵃᶜᵃ(i, j, k, grid, grid_marked_value, V) ≈ 10 * ℑyTᵃᶜᵃ(i, j, k, grid, V)
            @test ∂xTᶠᶜᶠ(i, j, k, grid, grid_marked_value, c) ≈ 10 * ∂xTᶠᶜᶠ(i, j, k, grid, c)
            @test ∂yTᶜᶠᶠ(i, j, k, grid, grid_marked_value, c) ≈ 10 * ∂yTᶜᶠᶠ(i, j, k, grid, c)
            @test δxTᶠᵃᵃ(i, j, k, underlying, grid_marked_value, c) ≈ δxTᶠᵃᵃ(i, j, k, underlying, c)
        end
    end

    @testset "Topology-aware interpolations" begin
        for topology in ((Periodic, Periodic, Bounded), (Bounded, Bounded, Bounded))
            grid = RectilinearGrid(CPU(), size=(8, 6, 2), x=(0, 1), y=(0, 1), z=(-1, 0); topology)
            Nx, Ny, Nz = size(grid)
            TX, TY, _ = topology

            c = randomize_interior_and_poison_halos!(CenterField(grid))
            U = randomize_interior_and_poison_halos!(XFaceField(grid))
            V = randomize_interior_and_poison_halos!(YFaceField(grid))

            # In the interior, the topology-aware interpolations equal the plain ones
            @test all(ℑxTᶠᵃᵃ(i, j, k, grid, c) ≈ ℑxᶠᵃᵃ(i, j, k, grid, c) for i in 2:Nx, j in 1:Ny, k in 1:Nz)
            @test all(ℑyTᵃᶠᵃ(i, j, k, grid, c) ≈ ℑyᵃᶠᵃ(i, j, k, grid, c) for i in 1:Nx, j in 2:Ny, k in 1:Nz)
            @test all(ℑxTᶜᵃᵃ(i, j, k, grid, U) ≈ ℑxᶜᵃᵃ(i, j, k, grid, U) for i in 1:Nx-1, j in 1:Ny, k in 1:Nz)
            @test all(ℑyTᵃᶜᵃ(i, j, k, grid, V) ≈ ℑyᵃᶜᵃ(i, j, k, grid, V) for i in 1:Nx, j in 1:Ny-1, k in 1:Nz)

            if TX === Periodic
                # Wrap around at both ends without reading the poisoned halos
                @test all(ℑxTᶠᵃᵃ(1, j, k, grid, c) ≈ (c[1, j, k] + c[Nx, j, k]) / 2 for j in 1:Ny, k in 1:Nz)
                @test all(ℑxTᶜᵃᵃ(Nx, j, k, grid, U) ≈ (U[Nx, j, k] + U[1, j, k]) / 2 for j in 1:Ny, k in 1:Nz)
                @test all(ℑyTᵃᶠᵃ(i, 1, k, grid, c) ≈ (c[i, 1, k] + c[i, Ny, k]) / 2 for i in 1:Nx, k in 1:Nz)
                @test all(ℑyTᵃᶜᵃ(i, Ny, k, grid, V) ≈ (V[i, Ny, k] + V[i, 1, k]) / 2 for i in 1:Nx, k in 1:Nz)
            else
                # Zero-gradient at the closed left wall, consistent with `δxTᶠᵃᵃ` vanishing there
                @test all(ℑxTᶠᵃᵃ(1, j, k, grid, c) == c[1, j, k] for j in 1:Ny, k in 1:Nz)
                @test all(ℑyTᵃᶠᵃ(i, 1, k, grid, c) == c[i, 1, k] for i in 1:Nx, k in 1:Nz)
                @test all(δxTᶠᵃᵃ(1, j, k, grid, c) == 0 for j in 1:Ny, k in 1:Nz)
                @test all(δyTᵃᶠᵃ(i, 1, k, grid, c) == 0 for i in 1:Nx, k in 1:Nz)

                # ... and at the closed right wall face (i = Nx + 1), beyond which the halos are poisoned
                @test all(ℑxTᶠᵃᵃ(Nx+1, j, k, grid, c) == c[Nx, j, k] for j in 1:Ny, k in 1:Nz)
                @test all(ℑyTᵃᶠᵃ(i, Ny+1, k, grid, c) == c[i, Ny, k] for i in 1:Nx, k in 1:Nz)
                @test all(δxTᶠᵃᵃ(Nx+1, j, k, grid, c) == 0 for j in 1:Ny, k in 1:Nz)
                @test all(δyTᵃᶠᵃ(i, Ny+1, k, grid, c) == 0 for i in 1:Nx, k in 1:Nz)
                @test all(ℑxTᶠᵃᵃ(Nx+1, j, k, grid, grid_marked_value, c) == c[Nx, j, k] for j in 1:Ny, k in 1:Nz)
                @test all(ℑyTᵃᶠᵃ(i, Ny+1, k, grid, grid_marked_value, c) == c[i, Ny, k] for i in 1:Nx, k in 1:Nz)
                @test all(δxTᶠᵃᵃ(Nx+1, j, k, grid, grid_marked_value, c) == 0 for j in 1:Ny, k in 1:Nz)
                @test all(δyTᵃᶠᵃ(i, Ny+1, k, grid, grid_marked_value, c) == 0 for i in 1:Nx, k in 1:Nz)

                # Interior faces are unchanged by the wall branches
                @test all(δxTᶠᵃᵃ(i, j, k, grid, c) ≈ δxᶠᵃᵃ(i, j, k, grid, c) for i in 2:Nx, j in 1:Ny, k in 1:Nz)
                @test all(δyTᵃᶠᵃ(i, j, k, grid, c) ≈ δyᵃᶠᵃ(i, j, k, grid, c) for i in 1:Nx, j in 2:Ny, k in 1:Nz)

                # The right-wall face is interior data, so the center interpolation reads it as usual
                @test all(ℑxTᶜᵃᵃ(Nx, j, k, grid, U) ≈ (U[Nx, j, k] + U[Nx+1, j, k]) / 2 for j in 1:Ny, k in 1:Nz)
                @test all(ℑyTᵃᶜᵃ(i, Ny, k, grid, V) ≈ (V[i, Ny, k] + V[i, Ny+1, k]) / 2 for i in 1:Nx, k in 1:Nz)
            end

            # Function arguments follow the same rules as arrays
            @test all(ℑxTᶠᵃᵃ(i, j, k, grid, grid_marked_value, c) ≈ ℑxTᶠᵃᵃ(i, j, k, grid, c) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test all(ℑyTᵃᶠᵃ(i, j, k, grid, grid_marked_value, c) ≈ ℑyTᵃᶠᵃ(i, j, k, grid, c) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test all(ℑxTᶜᵃᵃ(i, j, k, grid, grid_marked_value, U) ≈ ℑxTᶜᵃᵃ(i, j, k, grid, U) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
            @test all(ℑyTᵃᶜᵃ(i, j, k, grid, grid_marked_value, V) ≈ ℑyTᵃᶜᵃ(i, j, k, grid, V) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        end
    end
end
