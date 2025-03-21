module Models

import Oceananigans.Models: initialization_update_state!
import Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos

using ..TimeSteppers: ReactantModel
using ..Grids: ReactantGrid

initialization_update_state!(::ReactantModel; kw...) = nothing

# We may need this but not sure:
# maybe_extend_halos(TX, TY, ::ReactantGrid, ::FixedSubstepNumber) = grid

end # module
