using Oceananigans: @hascuda, CPU, GPU

#####
##### Utilities that make it easier to juggle around Arrays and CuArrays.
#####

         array_type(::CPU) = Array
@hascuda array_type(::GPU) = CuArray
