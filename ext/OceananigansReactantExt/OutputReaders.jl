module OutputReaders

import Oceananigans.OutputReaders: find_time_index

import Reactant.TracedRNumberOverrides: TracedStepRangeLen

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


end
