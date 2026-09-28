# Writes bottom_height.jld2: 1° block means of ETOPO 2022 (60 arc-second surface, sampled every 20 points) interpolated
# on the 1° tripolar grid of setup.jl. Usage: julia --project bottom_height.jl <path to ETOPO_2022_v1_60s_N90W180_surface.nc>

using Oceananigans
using Oceananigans.Fields: interpolate!
using NCDatasets, JLD2

etopo_filename = ARGS[1]
stride = 20
etopo_elevation = NCDataset(ds -> nomissing(ds["z"][stride÷2:stride:end, stride÷2:stride:end]), etopo_filename)

block_mean(a, n) = [sum(@view a[i:i+n-1, j:j+n-1]) / n^2 for i in 1:n:size(a, 1), j in 1:n:size(a, 2)]
elevation = Float64.(block_mean(etopo_elevation, 3))

z = MutableVerticalDiscretization([-4000, -1500, -500, -100, 0])
underlying_grid = TripolarGrid(CPU(), Float64; size=(360, 180, 4), z, halo=(5, 5, 5))
etopo_grid = LatitudeLongitudeGrid(CPU(), Float64; size=size(elevation), longitude=(-180, 180), latitude=(-90, 90),
                                   topology=(Periodic, Bounded, Flat))
elevation_field = CenterField(etopo_grid)
set!(elevation_field, elevation)
bottom_height = Field{Center, Center, Nothing}(underlying_grid)
interpolate!(bottom_height, elevation_field)

jldsave(joinpath(@__DIR__, "bottom_height.jld2"); bottom_height=Array(interior(bottom_height, :, :, 1)))
