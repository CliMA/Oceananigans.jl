module OutputReaders

using Oceananigans.Architectures: ReactantState
using Reactant: TracedStepRangeLen
using Oceananigans.OutputReaders: TimeInterpolator
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
    cpu_times = on_architecture(Oceananigans.Architectures.CPU(), times)
    return TimeInterpolator(time_indexing, times, t)
end

end
