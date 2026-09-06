using KernelAbstractions: @kernel, @index

using Oceananigans: location
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!, FieldBoundaryConditions
using Oceananigans.Fields: interior
using Oceananigans.Utils: launch!

# Each cell renormalizes over its own valid samples.
@kernel function _accumulate_sample!(total, weight, sample, overlap)
    i, j, k = @index(Global, NTuple)
    FT = eltype(total)
    @inbounds begin
        value = sample[i, j, k]
        valid = !isnan(value)
        total[i, j, k] += ifelse(valid, overlap * value, zero(FT))
        weight[i, j, k] += ifelse(valid, overlap, zero(FT))
    end
end

@kernel function _finalize_window!(averaged, total, weight, w)
    i, j, k = @index(Global, NTuple)
    FT = eltype(averaged)
    @inbounds averaged[i, j, k, w] = ifelse(weight[i, j, k] > 0,
                                            total[i, j, k] / weight[i, j, k],
                                            convert(FT, NaN))
end

# TODO: make this a lazy `FieldTimeSeriesOperation` once #5761 lands.
"""
$(TYPEDSIGNATURES)

Average `fts` onto consecutive windows of length `window`, in the units of `fts.times`.

`bounds` has `length(fts.times) + 1` entries and sample `n` covers `[bounds[n], bounds[n+1])`.
The windows tile `[first(bounds), last(bounds))` from `first(bounds)`, the last one truncated at
`last(bounds)` with a warning, and each sample is weighted by its overlap with the window. `NaN`
samples are skipped and the remaining weights renormalized, so a window with no valid sample is `NaN`.

The result is a `FieldTimeSeries` with the grid, indices, and time indexing of `fts` and
times at the window centers. Samples are read one at a time through `fts[n]`, so `fts` may
keep only part of its record in memory.

Example
=======

Average four daily samples onto two-day windows:

```jldoctest
using Oceananigans
using Oceananigans.Units

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
daily = FieldTimeSeries{Center, Center, Center}(grid, (0.5:3.5) * days)

sample = CenterField(grid)
for n in 1:4
    set!(sample, n)
    set!(daily, sample, n)
end

bounds = (0:4) * days # sample n covers [bounds[n], bounds[n+1])
averaged = time_average(daily, bounds, 2days)
Array(interior(averaged, 1, 1, 1, :))

# output
2-element Vector{Float64}:
 1.5
 3.5
```
"""
function time_average(fts::FieldTimeSeries, bounds, window)
    Nt = length(fts.times)
    length(bounds) == Nt + 1 ||
        throw(ArgumentError("bounds must have $(Nt + 1) entries, found $(length(bounds))"))

    edges = collect(first(bounds):window:last(bounds))
    if last(edges) < last(bounds)
        push!(edges, last(bounds))
        @warn "The last window spans $(last(bounds) - edges[end - 1]) rather than $window"
    end
    Nw = length(edges) - 1
    times = [(edges[w] + edges[w + 1]) / 2 for w in 1:Nw]

    boundary_conditions = FieldBoundaryConditions(fts.indices, fts.boundary_conditions)
    LX, LY, LZ = location(fts)
    output = FieldTimeSeries{LX, LY, LZ}(fts.grid, times; indices = fts.indices,
                                         time_indexing = fts.time_indexing,
                                         boundary_conditions)

    grid = fts.grid
    arch = architecture(grid)
    FT = eltype(grid)
    averaged = interior(output)
    spatial_size = size(averaged)[1:3]
    total = zeros(arch, FT, spatial_size...)
    weight = zeros(arch, FT, spatial_size...)

    for w in 1:Nw
        fill!(total, 0)
        fill!(weight, 0)

        for n in 1:Nt
            overlap = min(bounds[n + 1], edges[w + 1]) - max(bounds[n], edges[w])
            overlap > 0 || continue
            launch!(arch, grid, spatial_size, _accumulate_sample!,
                    total, weight, interior(fts[n]), convert(FT, overlap))
        end

        launch!(arch, grid, spatial_size, _finalize_window!, averaged, total, weight, w)
    end

    fill_halo_regions!(output)

    return output
end
