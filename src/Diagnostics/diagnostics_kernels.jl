#####
##### Useful kernels for doing diagnostics
#####

using KernelAbstractions

@kernel function velocity_div!(div, grid, u, v, w)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end
