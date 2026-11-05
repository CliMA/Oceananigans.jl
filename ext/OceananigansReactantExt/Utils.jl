module Utils

using Oceananigans
using Reactant

using Oceananigans.Architectures: ReactantState
using Oceananigans.Utils: prettytime, prettysummary

# Reactant tracing breaks if Δt is converted outside the kernel, so pass it through.
@inline Oceananigans.Utils.kernel_time_step(::ReactantState, grid, Δt) = Δt

function Oceananigans.Utils.prettytime(concrete_number::Union{ConcretePJRTNumber,ConcreteIFRTNumber})
    number = Reactant.to_number(concrete_number)
    return prettytime(number)
end

Oceananigans.Utils.prettytime(t::Reactant.TracedRNumber) = "TracedRNumber"

function Oceananigans.Utils.prettysummary(concrete_number::ConcretePJRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcretePJRTNumber(", prettysummary(number), ")")
end

function Oceananigans.Utils.prettysummary(concrete_number::ConcreteIFRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcreteIFRTNumber(", prettysummary(number), ")")
end

end # module
