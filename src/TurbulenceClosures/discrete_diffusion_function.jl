using Oceananigans.Operators: interpolation_operator

# To allow indexing a diffusivity with (i, j, k, grid, Lx, Ly, Lz)
# struct DiscreteDiffusionFunction{LX, LY, LZ, F, P} 

struct DiscreteDiffusionFunction{F} 
    func :: F
end
