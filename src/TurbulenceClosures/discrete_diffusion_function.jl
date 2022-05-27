using Oceananigans.Operators: interpolation_operator

# To allow indexing a diffusivity with (i, j, k, grid, Lx, Ly, Lz)
struct DiscreteDiffusionFunction{F} 
    func :: F
end

# function DiscreteDiffusionFunction(func, location)
#     if isnothing(location)
#         return DiscreteDiffusionFunction(func)
#     else
#         function new_func(i, j, k, grid, lx, ly, lz, files)
#             ℑ = interpolation_operator(location, (lx, ly, lz))

#             return ℑ(i, j, k, grid, func, Center(), Center(), Center(), files)
#         end
#         return DiscreteDiffusionFunction(new_func)
#     end
# end
    
