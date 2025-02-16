module TimeSteppers

using Reactant
using Oceananigans

using Oceananigans.Grids: AbstractGrid
using ..Architectures: ReactantState

import Oceananigans.TimeSteppers: Clock

const ReactantGrid{FT, TX, TY, TZ} = AbstractGrid{FT, TX, TY, TZ, <:ReactantState} where {FT, TX, TY, TZ}

function Clock(grid::ReactantGrid)
    FT = Float64 # may change in the future
    t = ConcreteRNumber(zero(FT))
    iter = ConcreteRNumber(0)
    stage = ConcreteRNumber(0)
    last_Δt = ConcreteRNumber(zero(FT))
    last_stage_Δt = ConcreteRNumber(zero(FT))
    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

end # module

