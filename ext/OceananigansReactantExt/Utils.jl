module Utils

using Oceananigans
using Reactant

import Oceananigans.Utils: prettytime, prettysummary, kernel_time_step

using Oceananigans.Architectures: ReactantState

# Reactant tracing breaks if Δt is converted outside the kernel, so pass it through.
@inline kernel_time_step(::ReactantState, grid, Δt) = Δt

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
