module ConvergenceTests

using JLD2, Statistics

using Oceananigans, Oceananigans.Fields

import Oceananigans: RegularCartesianGrid

include("file_wrangling.jl")
include("analysis.jl")

include("ForcedFlowFreeSlip.jl")
include("GaussianAdvectionDiffusion.jl")

end
