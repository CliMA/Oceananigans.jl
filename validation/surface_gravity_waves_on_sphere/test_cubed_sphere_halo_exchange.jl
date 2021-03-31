using Logging
using Printf
using Test
using DataDeps

using Oceananigans
using Oceananigans.CubedSpheres

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.CubedSpheres: west_halo, east_halo, south_halo, north_halo

# Opposite of the `digits` function
# Source: https://stackoverflow.com/a/55529778
function undigits(d; base=10)
    (s, b) = promote(zero(eltype(d)), base)
    mult = one(s)
    for val in d
        s += val * mult
        mult *= b
    end
    return s
end

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

Logging.global_logger(OceananigansLogger())

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "3cc5d86290c3af028cddfa47e61e095ee470fe6f8d779c845de09da2f1abeb15" # sha256sum
)

DataDeps.register(dd)
cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

@testset "Cubed sphere halo exchange" begin
    arch = CPU()
    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))
    field = CenterField(Float64, arch, grid)

    ## We will fill each grid point with a 5-digit integer fiijj where
    ## the f digit is the face number, the ii digits are the i index, and
    ## the jj digits are the j index. We then check that the halo exchange
    ## happened correctly.

    face_digit(n) = digits(Int(n))[5]
    i_digits(n) = digits(Int(n))[3:4] |> undigits
    j_digits(n) = digits(Int(n))[1:2] |> undigits

    for (face_number, field_face) in enumerate(field.faces)
        for i in 1:field_face.grid.Nx, j in 1:field_face.grid.Ny
            field_face[i, j, 1] = parse(Int, @sprintf("%d%02d%02d", face_number, i, j))
        end
    end

    fill_halo_regions!(field, arch)

    @testset "Source and destination faces are correct" begin
        for (face_number, field_face) in enumerate(field.faces)
            west_halo_vals =  west_halo(field_face, include_corners=false)
            east_halo_vals =  east_halo(field_face, include_corners=false)
            south_halo_vals = south_halo(field_face, include_corners=false)
            north_halo_vals = north_halo(field_face, include_corners=false)

            @test all(face_digit.(west_halo_vals)  .== grid.face_connectivity[face_number].west.face)
            @test all(face_digit.(east_halo_vals)  .== grid.face_connectivity[face_number].east.face)
            @test all(face_digit.(south_halo_vals) .== grid.face_connectivity[face_number].south.face)
            @test all(face_digit.(north_halo_vals) .== grid.face_connectivity[face_number].north.face)
        end
    end

    ## Test 1W halo <- 5N boundary halo exchange

    @testset "1W halo <- 5N boundary halo exchange" begin
        # Grid point (i, j) = (0, 1) in 1W halo should be from (i, j) = (32, 32) in 5N boundary.
        west_halo_south_value = field.faces[1][0, 1, 1]
        @test i_digits(west_halo_south_value) == 32
        @test j_digits(west_halo_south_value) == 32

        # Grid point (i, j) = (0, 32) in 1W halo should be from (i, j) = (1, 32) in 5N boundary.
        west_halo_north_value = field.faces[1][0, 32, 1]
        @test i_digits(west_halo_north_value) == 1
        @test j_digits(west_halo_north_value) == 32

        @test i_digits.(west_halo(field.faces[1], include_corners=false)) == reverse(1:32)
        @test all(j_digits.(west_halo(field.faces[1], include_corners=false))[:] .== 32)
    end
end
