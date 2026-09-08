module OutputReaders

using GPUArraysCore: @allowscalar
using Oceananigans.Architectures: ReactantState, on_architecture, CPU, architecture
using Oceananigans.Fields: Field, FixedTime, instantiated_location
using Oceananigans.Grids: offset_data
using Oceananigans.OutputReaders: TimeInterpolator, TotallyInMemoryFTS, memory_index
using Oceananigans.Units: Time
using Reactant: TracedStepRangeLen, TracedRNumber
import Oceananigans.OutputReaders: find_time_index, cpu_interpolating_time_indices

@inline function find_time_index(times::TracedStepRangeLen, t)
    n₂ = searchsortedfirst(times, t)

    Nt = length(times)
    n₂ = min(Nt, n₂) # cap
    n₁ = max(1, n₂ - 1)

    @inbounds begin
        t₁ = times[n₁]
        t₂ = times[n₂]
    end

    ñ = (t - t₁) / (t₂ - t₁)
    ñ = ifelse(n₂ == n₁, zero(ñ), ñ)

    return ñ, n₁, n₂
end

function cpu_interpolating_time_indices(::ReactantState, times, time_indexing, t)
    cpu_times = on_architecture(CPU(), times)
    return @allowscalar TimeInterpolator(time_indexing, cpu_times, t)
end

# A dynamic slice along the time axis, so the index may be known only at run time
snapshot(fts, n) = parent(fts)[:, :, :, memory_index(fts, n)]

function Base.getindex(fts::TotallyInMemoryFTS, n::TracedRNumber)
    loc = instantiated_location(fts)
    data = offset_data(snapshot(fts, n), fts.grid, loc, fts.indices)
    status = @allowscalar FixedTime(fts.times[n])
    return Field(loc, fts.grid; data, fts.boundary_conditions, fts.indices, status)
end

function Base.getindex(fts::TotallyInMemoryFTS, time_index::Time{<:TracedRNumber})
    indices = cpu_interpolating_time_indices(architecture(fts), fts.times, fts.time_indexing, time_index.time)

    # `ñ = 0` when `n₁ == n₂`, so no branch is needed
    ñ  = TracedRNumber{eltype(fts.grid)}(indices.fractional_index)
    ψ₁ = snapshot(fts, indices.first_index)
    ψ₂ = snapshot(fts, indices.second_index)

    loc = instantiated_location(fts)
    data = offset_data(@.(ψ₂ * ñ + ψ₁ * (1 - ñ)), fts.grid, loc, fts.indices)
    status = FixedTime(time_index.time)

    return Field(loc, fts.grid; data, fts.boundary_conditions, fts.indices, status)
end

end
