using Oceananigans
using Oceananigans.Advection: RotatedAdvection

include("compute_rpe.jl")
include("baroclinic_adjustment.jl")

momentum_advection = WENOVectorInvariant()

w5 = WENO(; order = 5)
w7 = WENO(; order = 7)

tracer_advections  = [
    w5,
    w7,
    RotatedAdvection(w5),
    RotatedAdvection(w7)
]

filenames = [
    "baroclinic_adjustment_weno5",
    "baroclinic_adjustment_weno7",
    "baroclinic_adjustment_rotated_weno5",
    "baroclinic_adjustment_rotated_weno7"
]

sim = baroclinic_adjustment_simulation(1/6, filenames[1]; 
                                       arch = GPU(), 
                                       momentum_advection,
                                       tracer_advection = tracer_advections[1])
run!(sim)

# sim = baroclinic_adjustment_simulation(1/6, filenames[2]; 
#                                        arch = GPU(), 
#                                        momentum_advection,
#                                        tracer_advection = tracer_advections[2])
# run!(sim)


sim = baroclinic_adjustment_simulation(1/6, filenames[3]; 
                                       arch = GPU(), 
                                       momentum_advection,
                                       tracer_advection = tracer_advections[3])
run!(sim)

# sim = baroclinic_adjustment_simulation(1/6, filenames[4]; 
#                                        arch = GPU(), 
#                                        momentum_advection,
#                                        tracer_advection = tracer_advections[4])
# run!(sim)

