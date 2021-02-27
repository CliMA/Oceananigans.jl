using Test

include("conformal_cubed_sphere_grid.jl")

function run_cubed_sphere_face_array_size_tests()
    grid = ConformalCubedSphereFaceGrid(size=(10, 10, 1), z=(0, 1))

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    @test grid.λᶜᶜᶜ isa OffsetArray{Float64, 2, <:Array}
    @test grid.λᶠᶜᶜ isa OffsetArray{Float64, 2, <:Array}
    @test grid.λᶜᶠᶜ isa OffsetArray{Float64, 2, <:Array}
    @test grid.λᶠᶠᶜ isa OffsetArray{Float64, 2, <:Array}

    @test grid.ϕᶜᶜᶜ isa OffsetArray{Float64, 2, <:Array}
    @test grid.ϕᶠᶜᶜ isa OffsetArray{Float64, 2, <:Array}
    @test grid.ϕᶜᶠᶜ isa OffsetArray{Float64, 2, <:Array}
    @test grid.ϕᶠᶠᶜ isa OffsetArray{Float64, 2, <:Array}

    @test size(grid.λᶜᶜᶜ) == (Nx + 2Hx,     Ny + 2Hy    )
    @test size(grid.λᶠᶜᶜ) == (Nx + 2Hx + 1, Ny + 2Hy    )
    @test size(grid.λᶜᶠᶜ) == (Nx + 2Hx,     Ny + 2Hy + 1)
    @test size(grid.λᶠᶠᶜ) == (Nx + 2Hx + 1, Ny + 2Hy + 1)

    @test size(grid.ϕᶜᶜᶜ) == (Nx + 2Hx,     Ny + 2Hy    )
    @test size(grid.ϕᶠᶜᶜ) == (Nx + 2Hx + 1, Ny + 2Hy    )
    @test size(grid.ϕᶜᶠᶜ) == (Nx + 2Hx,     Ny + 2Hy + 1)
    @test size(grid.ϕᶠᶠᶜ) == (Nx + 2Hx + 1, Ny + 2Hy + 1)

    return nothing
end

@testset "Cubed sphere grid" begin
    run_cubed_sphere_face_array_size_tests()
end
