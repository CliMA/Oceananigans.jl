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

    ## We will fill each grid point with a 5-digit integer "fiijj" where
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

    @testset "1W halo <- 5N boundary halo exchange" begin
        # Grid point (i, j) = (0, 1) in 1W halo should be from (i, j) = (32, 32) in 5N boundary.
        west_halo_south_value = field.faces[1][0, 1, 1]
        @test face_digit(west_halo_south_value) == 5
        @test i_digits(west_halo_south_value) == 32
        @test j_digits(west_halo_south_value) == 32

        # Grid point (i, j) = (0, 32) in 1W halo should be from (i, j) = (1, 32) in 5N boundary.
        west_halo_north_value = field.faces[1][0, 32, 1]
        @test face_digit(west_halo_north_value) == 5
        @test i_digits(west_halo_north_value) == 1
        @test j_digits(west_halo_north_value) == 32

        west_halo_values = west_halo(field.faces[1], include_corners=false)[:]
        @test all(face_digit.(west_halo_values) .== 5)
        @test all(i_digits.(west_halo_values) .== reverse(1:32))
        @test all(j_digits.(west_halo_values) .== 32)
    end

    @testset "1E halo <- 2W boundary halo exchange" begin
        # Grid point (i, j) = (33, 1) in 1E halo should be from (i, j) = (1, 1) in 2W boundary.
        east_halo_south_value = field.faces[1][33, 1, 1]
        @test face_digit(east_halo_south_value) == 2
        @test i_digits(east_halo_south_value) == 1
        @test j_digits(east_halo_south_value) == 1

        # Grid point (i, j) = (33, 32) in 1E halo should be from (i, j) = (1, 32) in 2W boundary.
        east_halo_north_value = field.faces[1][33, 32, 1]
        @test face_digit(east_halo_north_value) == 2
        @test i_digits(east_halo_north_value) == 1
        @test j_digits(east_halo_north_value) == 32

        east_halo_values = east_halo(field.faces[1], include_corners=false)[:]
        @test all(face_digit.(east_halo_values) .== 2)
        @test all(i_digits.(east_halo_values) .== 1)
        @test all(j_digits.(east_halo_values) .== 1:32)
    end

    @testset "1S halo <- 6N boundary halo exchange" begin
        # Grid point (i, j) = (1, 0) in 1S halo should be from (i, j) = (1, 32) in 6N boundary.
        south_halo_west_value = field.faces[1][1, 0, 1]
        @test face_digit(south_halo_west_value) == 6
        @test i_digits(south_halo_west_value) == 1
        @test j_digits(south_halo_west_value) == 32

        # Grid point (i, j) = (32, 0) in 1S halo should be from (i, j) = (32, 32) in 6N boundary.
        south_halo_east_value = field.faces[1][32, 0, 1]
        @test face_digit(south_halo_east_value) == 6
        @test i_digits(south_halo_east_value) == 32
        @test j_digits(south_halo_east_value) == 32

        south_halo_values = south_halo(field.faces[1], include_corners=false)[:]
        @test all(face_digit.(south_halo_values) .== 6)
        @test all(i_digits.(south_halo_values) .== 1:32)
        @test all(j_digits.(south_halo_values) .== 32)
    end

    @testset "1N halo <- 3W boundary halo exchange" begin
        # Grid point (i, j) = (1, 33) in 1N halo should be from (i, j) = (1, 32) in 3W boundary.
        north_halo_west_value = field.faces[1][1, 33, 1]
        @test face_digit(north_halo_west_value) == 3
        @test i_digits(north_halo_west_value) == 1
        @test j_digits(north_halo_west_value) == 32

        # Grid point (i, j) = (32, 33) in 1N halo should be from (i, j) = (1, 1) in 3W boundary.
        north_halo_east_value = field.faces[1][32, 33, 1]
        @test face_digit(north_halo_east_value) == 3
        @test i_digits(north_halo_east_value) == 1
        @test j_digits(north_halo_east_value) == 1

        north_halo_values = north_halo(field.faces[1], include_corners=false)[:]
        @test all(face_digit.(north_halo_values) .== 3)
        @test all(i_digits.(north_halo_values) .== 1)
        @test all(j_digits.(north_halo_values) .== reverse(1:32))
    end

    @testset "2W halo <- 1E boundary halo exchange" begin
        # Grid point (i, j) = (0, 1) in 2W halo should be from (i, j) = (32, 1) in 1E boundary.
        west_halo_south_value = field.faces[2][0, 1, 1]
        @test face_digit(west_halo_south_value) == 1
        @test i_digits(west_halo_south_value) == 32
        @test j_digits(west_halo_south_value) == 1

        # Grid point (i, j) = (0, 32) in 2W halo should be from (i, j) = (32, 32) in 1E boundary.
        west_halo_north_value = field.faces[2][0, 32, 1]
        @test face_digit(west_halo_north_value) == 1
        @test i_digits(west_halo_north_value) == 32
        @test j_digits(west_halo_north_value) == 32

        west_halo_values = west_halo(field.faces[2], include_corners=false)[:]
        @test all(face_digit.(west_halo_values) .== 1)
        @test all(i_digits.(west_halo_values) .== 32)
        @test all(j_digits.(west_halo_values) .== 1:32)
    end

    @testset "2E halo <- 4S boundary halo exchange" begin
        # Grid point (i, j) = (33, 1) in 2E halo should be from (i, j) = (32, 1) in 4S boundary.
        east_halo_south_value = field.faces[2][33, 1, 1]
        @test face_digit(east_halo_south_value) == 4
        @test i_digits(east_halo_south_value) == 32
        @test j_digits(east_halo_south_value) == 1

        # Grid point (i, j) = (33, 32) in 2E halo should be from (i, j) = (1, 1) in 4S boundary.
        east_halo_north_value = field.faces[2][33, 32, 1]
        @test face_digit(east_halo_north_value) == 4
        @test i_digits(east_halo_north_value) == 1
        @test j_digits(east_halo_north_value) == 1

        east_halo_values = east_halo(field.faces[2], include_corners=false)[:]
        @test all(face_digit.(east_halo_values) .== 4)
        @test all(i_digits.(east_halo_values) .== reverse(1:32))
        @test all(j_digits.(east_halo_values) .== 1)
    end

    @testset "2S halo <- 6E boundary halo exchange" begin
        # Grid point (i, j) = (1, 0) in 2S halo should be from (i, j) = (32, 32) in 6E boundary.
        south_halo_west_value = field.faces[2][1, 0, 1]
        @test face_digit(south_halo_west_value) == 6
        @test i_digits(south_halo_west_value) == 32
        @test j_digits(south_halo_west_value) == 32

        # Grid point (i, j) = (32, 0) in 2S halo should be from (i, j) = (32, 1) in 6E boundary.
        south_halo_east_value = field.faces[2][32, 0, 1]
        @test face_digit(south_halo_east_value) == 6
        @test i_digits(south_halo_east_value) == 32
        @test j_digits(south_halo_east_value) == 1

        south_halo_values = south_halo(field.faces[2], include_corners=false)[:]
        @test all(face_digit.(south_halo_values) .== 6)
        @test all(i_digits.(south_halo_values) .== 32)
        @test all(j_digits.(south_halo_values) .== reverse(1:32))
    end

    @testset "2N halo <- 3S boundary halo exchange" begin
        # Grid point (i, j) = (1, 33) in 2N halo should be from (i, j) = (1, 1) in 3S boundary.
        north_halo_west_value = field.faces[2][1, 33, 1]
        @test face_digit(north_halo_west_value) == 3
        @test i_digits(north_halo_west_value) == 1
        @test j_digits(north_halo_west_value) == 1

        # Grid point (i, j) = (32, 33) in 2N halo should be from (i, j) = (32, 1) in 3S boundary.
        north_halo_east_value = field.faces[2][32, 33, 1]
        @test face_digit(north_halo_east_value) == 3
        @test i_digits(north_halo_east_value) == 32
        @test j_digits(north_halo_east_value) == 1

        north_halo_values = north_halo(field.faces[2], include_corners=false)[:]
        @test all(face_digit.(north_halo_values) .== 3)
        @test all(i_digits.(north_halo_values) .== 1:32)
        @test all(j_digits.(north_halo_values) .== 1)
    end

    @testset "3W halo <- 1N boundary halo exchange" begin
        # Grid point (i, j) = (0, 1) in 3W halo should be from (i, j) = (32, 32) in 1N boundary.
        west_halo_south_value = field.faces[3][0, 1, 1]
        @test face_digit(west_halo_south_value) == 1
        @test i_digits(west_halo_south_value) == 32
        @test j_digits(west_halo_south_value) == 32

        # Grid point (i, j) = (0, 32) in 3W halo should be from (i, j) = (1, 32) in 1N boundary.
        west_halo_north_value = field.faces[3][0, 32, 1]
        @test face_digit(west_halo_north_value) == 1
        @test i_digits(west_halo_north_value) == 1
        @test j_digits(west_halo_north_value) == 32

        west_halo_values = west_halo(field.faces[3], include_corners=false)[:]
        @test all(face_digit.(west_halo_values) .== 1)
        @test all(i_digits.(west_halo_values) .== reverse(1:32))
        @test all(j_digits.(west_halo_values) .== 32)
    end

    @testset "3E halo <- 4W boundary halo exchange" begin
        # Grid point (i, j) = (33, 1) in 3E halo should be from (i, j) = (1, 1) in 4W boundary.
        east_halo_south_value = field.faces[3][33, 1, 1]
        @test face_digit(east_halo_south_value) == 4
        @test i_digits(east_halo_south_value) == 1
        @test j_digits(east_halo_south_value) == 1

        # Grid point (i, j) = (33, 32) in 3E halo should be from (i, j) = (1, 32) in 4W boundary.
        east_halo_north_value = field.faces[3][33, 32, 1]
        @test face_digit(east_halo_north_value) == 4
        @test i_digits(east_halo_north_value) == 1
        @test j_digits(east_halo_north_value) == 32

        east_halo_values = east_halo(field.faces[3], include_corners=false)[:]
        @test all(face_digit.(east_halo_values) .== 4)
        @test all(i_digits.(east_halo_values) .== 1)
        @test all(j_digits.(east_halo_values) .== 1:32)
    end

    @testset "3S halo <- 2N boundary halo exchange" begin
        # Grid point (i, j) = (1, 0) in 3S halo should be from (i, j) = (1, 32) in 2N boundary.
        south_halo_west_value = field.faces[3][1, 0, 1]
        @test face_digit(south_halo_west_value) == 2
        @test i_digits(south_halo_west_value) == 1
        @test j_digits(south_halo_west_value) == 32

        # Grid point (i, j) = (32, 0) in 3S halo should be from (i, j) = (32, 32) in 2N boundary.
        south_halo_east_value = field.faces[3][32, 0, 1]
        @test face_digit(south_halo_east_value) == 2
        @test i_digits(south_halo_east_value) == 32
        @test j_digits(south_halo_east_value) == 32

        south_halo_values = south_halo(field.faces[3], include_corners=false)[:]
        @test all(face_digit.(south_halo_values) .== 2)
        @test all(i_digits.(south_halo_values) .== 1:32)
        @test all(j_digits.(south_halo_values) .== 32)
    end

    @testset "3N halo <- 5W boundary halo exchange" begin
        # Grid point (i, j) = (1, 33) in 3N halo should be from (i, j) = (1, 32) in 5W boundary.
        north_halo_west_value = field.faces[3][1, 33, 1]
        @test face_digit(north_halo_west_value) == 5
        @test i_digits(north_halo_west_value) == 1
        @test j_digits(north_halo_west_value) == 32

        # Grid point (i, j) = (32, 33) in 3N halo should be from (i, j) = (1, 1) in 5W boundary.
        north_halo_east_value = field.faces[3][32, 33, 1]
        @test face_digit(north_halo_east_value) == 5
        @test i_digits(north_halo_east_value) == 1
        @test j_digits(north_halo_east_value) == 1

        north_halo_values = north_halo(field.faces[3], include_corners=false)[:]
        @test all(face_digit.(north_halo_values) .== 5)
        @test all(i_digits.(north_halo_values) .== 1)
        @test all(j_digits.(north_halo_values) .== reverse(1:32))
    end

    @testset "4W halo <- 3E boundary halo exchange" begin
        # Grid point (i, j) = (0, 1) in 4W halo should be from (i, j) = (32, 1) in 3E boundary.
        west_halo_south_value = field.faces[4][0, 1, 1]
        @test face_digit(west_halo_south_value) == 3
        @test i_digits(west_halo_south_value) == 32
        @test j_digits(west_halo_south_value) == 1

        # Grid point (i, j) = (0, 32) in 4W halo should be from (i, j) = (32, 32) in 1N boundary.
        west_halo_north_value = field.faces[4][0, 32, 1]
        @test face_digit(west_halo_north_value) == 3
        @test i_digits(west_halo_north_value) == 32
        @test j_digits(west_halo_north_value) == 32

        west_halo_values = west_halo(field.faces[4], include_corners=false)[:]
        @test all(face_digit.(west_halo_values) .== 3)
        @test all(i_digits.(west_halo_values) .== 32)
        @test all(j_digits.(west_halo_values) .== 1:32)
    end

    @testset "4E halo <- 6S boundary halo exchange" begin
        # Grid point (i, j) = (33, 1) in 4E halo should be from (i, j) = (32, 1) in 6S boundary.
        east_halo_south_value = field.faces[4][33, 1, 1]
        @test face_digit(east_halo_south_value) == 6
        @test i_digits(east_halo_south_value) == 32
        @test j_digits(east_halo_south_value) == 1

        # Grid point (i, j) = (33, 32) in 4E halo should be from (i, j) = (1, 1) in 6S boundary.
        east_halo_north_value = field.faces[4][33, 32, 1]
        @test face_digit(east_halo_north_value) == 6
        @test i_digits(east_halo_north_value) == 1
        @test j_digits(east_halo_north_value) == 1

        east_halo_values = east_halo(field.faces[4], include_corners=false)[:]
        @test all(face_digit.(east_halo_values) .== 6)
        @test all(i_digits.(east_halo_values) .== reverse(1:32))
        @test all(j_digits.(east_halo_values) .== 1)
    end

    @testset "4S halo <- 2E boundary halo exchange" begin
        # Grid point (i, j) = (1, 0) in 4S halo should be from (i, j) = (32, 32) in 2E boundary.
        south_halo_west_value = field.faces[4][1, 0, 1]
        @test face_digit(south_halo_west_value) == 2
        @test i_digits(south_halo_west_value) == 32
        @test j_digits(south_halo_west_value) == 32

        # Grid point (i, j) = (32, 0) in 4S halo should be from (i, j) = (32, 1) in 2E boundary.
        south_halo_east_value = field.faces[4][32, 0, 1]
        @test face_digit(south_halo_east_value) == 2
        @test i_digits(south_halo_east_value) == 32
        @test j_digits(south_halo_east_value) == 1

        south_halo_values = south_halo(field.faces[4], include_corners=false)[:]
        @test all(face_digit.(south_halo_values) .== 2)
        @test all(i_digits.(south_halo_values) .== 32)
        @test all(j_digits.(south_halo_values) .== reverse(1:32))
    end

    @testset "4N halo <- 5S boundary halo exchange" begin
        # Grid point (i, j) = (1, 33) in 4N halo should be from (i, j) = (1, 1) in 5S boundary.
        north_halo_west_value = field.faces[4][1, 33, 1]
        @test face_digit(north_halo_west_value) == 5
        @test i_digits(north_halo_west_value) == 1
        @test j_digits(north_halo_west_value) == 1

        # Grid point (i, j) = (32, 33) in 4N halo should be from (i, j) = (32, 1) in 5S boundary.
        north_halo_east_value = field.faces[4][32, 33, 1]
        @test face_digit(north_halo_east_value) == 5
        @test i_digits(north_halo_east_value) == 32
        @test j_digits(north_halo_east_value) == 1

        north_halo_values = north_halo(field.faces[4], include_corners=false)[:]
        @test all(face_digit.(north_halo_values) .== 5)
        @test all(i_digits.(north_halo_values) .== 1:32)
        @test all(j_digits.(north_halo_values) .== 1)
    end

end
