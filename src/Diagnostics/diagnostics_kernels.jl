#####
##### Useful kernels for doing diagnostics
#####

using KernelAbstractions

@kernel function velocity_div!(grid, u, v, w, div)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end
