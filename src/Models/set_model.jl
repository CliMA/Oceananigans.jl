using Oceananigans.Utils

import Oceananigans: restore_prognostic_state!
import Oceananigans.Fields: set!

#####
##### set! for checkpointer filepaths and `OceananigansModels`
#####

"""
$(TYPEDSIGNATURES)

Restore `model` from checkpoint data stored at `filepath`.
"""
function set!(model::OceananigansModels, filepath::AbstractString)
    state = Oceananigans.OutputWriters.load_checkpoint_state(filepath; base_path="simulation/model")
    restore_prognostic_state!(model, state)
    return nothing
end
