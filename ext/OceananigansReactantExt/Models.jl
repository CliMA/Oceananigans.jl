module Models

import Oceananigans.Models: initialization_update_state!

using ..TimeSteppers: ReactantModel

initialization_update_state!(::ReactantModel; kw...) = nothing

end # module
