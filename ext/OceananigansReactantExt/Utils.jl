module Utils

using Oceananigans
using Reactant

import Oceananigans.Utils: prettysummary, prettytime

function prettytime(concrete_number::Union{ConcretePJRTNumber,ConcreteIFRTNumber})
    number = Reactant.to_number(concrete_number)
    return prettytime(number)
end

function prettysummary(concrete_number::ConcretePJRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcretePJRTNumber(", prettysummary(number), ")")
end

function prettysummary(concrete_number::ConcreteIFRTNumber)
    number = Reactant.to_number(concrete_number)
    return string("ConcreteIFRTNumber(", prettysummary(number), ")")
end

end # module
