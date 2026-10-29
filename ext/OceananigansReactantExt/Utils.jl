module Utils

using Oceananigans
using Reactant

import Oceananigans.Utils: prettytime, prettysummary, kernel_time_step, initialize_on_construction!

using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed

# Reactant tracing breaks if Δt is converted outside the kernel, so pass it through.
@inline kernel_time_step(::ReactantState, grid, Δt) = Δt

# Construction leaves the initialization to a compiled `initialize!(model)` (Models.jl)
initialize_on_construction!(::ReactantState, initialize!, args...) = nothing
initialize_on_construction!(::Distributed{<:ReactantState}, initialize!, args...) = nothing

function prettytime(concrete_number::Union{ConcretePJRTNumber,ConcreteIFRTNumber})
    number = Reactant.to_number(concrete_number)
    return prettytime(number)
end

prettytime(t::Reactant.TracedRNumber) = "TracedRNumber"

function prettysummary(concrete_number::ConcretePJRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcretePJRTNumber(", prettysummary(number), ")")
end

function prettysummary(concrete_number::ConcreteIFRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcreteIFRTNumber(", prettysummary(number), ")")
end

end # module
