# Generate pre-regridded bathymetry files for benchmarks.
#
# Run from the repository root:
#   julia --project=benchmarking benchmarking/generate_bathymetry.jl
#
# This script requires NumericalEarth.jl. After generating, upload the files
# and update BATHYMETRY_URL in benchmarking/src/OceananigansBenchmarks.jl.

using Oceananigans
using NumericalEarth
using JLD2

output_dir = joinpath(@__DIR__, "bathymetry")
mkpath(output_dir)

# All unique (grid_type, Nx, Ny) combinations used in benchmarks
configurations = [
    ("tripolar", 180, 90),
    ("tripolar", 360, 180),
    ("tripolar", 720, 360),
    ("latlon",   360, 180),
]

for (grid_type, Nx, Ny) in configurations
    println("Generating bathymetry for $grid_type $Nx×$Ny...")

    Nz = 10  # minimal; only horizontal coords matter for bathymetry
    if grid_type == "tripolar"
        grid = TripolarGrid(CPU();
            size = (Nx, Ny, Nz),
            halo = (7, 7, 7),
            z = (-5000, 0)
        )
    else
        grid = LatitudeLongitudeGrid(CPU();
            size = (Nx, Ny, Nz),
            halo = (7, 7, 7),
            longitude = (0, 360),
            latitude = (-80, 85),
            z = (-5000, 0)
        )
    end

    bottom_height = NumericalEarth.regrid_bathymetry(grid;
        minimum_depth = 10,
        interpolation_passes = 10,
        major_basins = 2
    )

    filename = "bathymetry_$(grid_type)_$(Nx)x$(Ny).jld2"
    filepath = joinpath(output_dir, filename)
    jldsave(filepath; bottom_height = Array(bottom_height))
    println("  Saved: $filepath ($(filesize(filepath)) bytes)")
end

println("\nAll bathymetry files generated in $output_dir")
println("Upload these files and update BATHYMETRY_URL in benchmarking/src/OceananigansBenchmarks.jl")
