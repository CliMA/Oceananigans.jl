module TurbulenceClosures

using Reactant
using .Models: ReactantModel

import Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: update_previous_compute_time!

update_previous_compute_time!(diffusivities, model::ReactantModel) = model.clock.last_Δt

end # module TurbulenceClosures

