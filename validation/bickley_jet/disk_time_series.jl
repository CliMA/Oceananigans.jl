module DiskTimeSerieses

using JLD2

using Oceananigans
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: offset_data

import Oceananigans.Grids: nodes

struct DiskTimeSeries{F, N, G, L, I, T}
    filepath :: F
    name :: N
    grid :: G
    location :: L
    iterations :: I
    times :: T
end

"""
    DiskTimeSeries(name, filepath)
Returns an abstraction for a time series of `Oceananigans.Field` data
stored on disk at `filepath` with `name`.
Example
=======
```
julia> using Oceananigans
julia> using GeophysicalDissipation.DiskTimeSerieses: DiskTimeSeries
julia> u_timeseries = DiskTimeSeries(:u, "pretty_cool_data.jld2")
julia> u_timeseries[i] isa AbstractField{Cell, Cell, Cell} # returns `u` at save point `i`
true
```
"""
function DiskTimeSeries(name, filepath)
    file = jldopen(filepath)

    grid = file["serialized/grid"]
    location = file["timeseries/$name/meta/location"]
    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$iter"] for iter in iterations]

    close(file)

    return DiskTimeSeries(filepath, name, grid, location, iterations, times)
end

nodes(dts::DiskTimeSeries) = nodes(dts.location, dts.grid)

function Base.getindex(dts::DiskTimeSeries, i)
    iter = dts.iterations[i]

    file = jldopen(dts.filepath)
    raw_data = file["timeseries/$(dts.name)/$iter"]
    close(file)

    return Field(dts.location,
                 CPU(),
                 dts.grid,
                 nothing, # boundary conditions
                 offset_data(raw_data, dts.grid, dts.location))
end

end # module
