# These kernels run on GPUs, some of which (e.g. Metal) cannot compile Float64
# arithmetic, so index computations must stay in integer arithmetic throughout:
# i ÷ 2 == trunc(i/2) and (N + 1) ÷ 2 == ceil(N/2) for the i, N ≥ 0 used here.

"""
$(TYPEDSIGNATURES)

Permute `i` such that, for example, `i ∈ 1:N` becomes

    [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 8, 2, 7, 3, 6, 4, 5]

    [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 9, 2, 8, 3, 7, 4, 6, 5]

for `N=8` and `N=9` respectively.

See equation (20) of [Makhoul80](@citet).
"""
@inline permute_index(i, N)::Int = ifelse(isodd(i),
                                          i ÷ 2 + 1,
                                          N - (i - 1) ÷ 2)

"""
$(TYPEDSIGNATURES)

Permute `i` in the opposite manner as `permute_index`, such that,
for example, `i ∈ 1:N` becomes

   [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 3, 5, 7, 8, 6, 4, 2]

   [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 3, 5, 7, 9, 8, 6, 4, 2]

for `N=8` and `N=9` respectively.

See equation (20) of [Makhoul80](@citet).
"""
@inline unpermute_index(i, N) = ifelse(i <= (N + 1) ÷ 2, 2i-1, 2(N-i+1))

@kernel function permute_x_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    i′ = permute_index(i, grid.Nx)
    @inbounds dst[i′, j, k] = src[i, j, k]
end

@kernel function permute_y_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    j′ = permute_index(j, grid.Ny)
    @inbounds dst[i, j′, k] = src[i, j, k]
end

@kernel function permute_z_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    k′ = permute_index(k, grid.Nz)
    @inbounds dst[i, j, k′] = src[i, j, k]
end

@kernel function unpermute_x_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    i′ = unpermute_index(i, grid.Nx)
    @inbounds dst[i′, j, k] = src[i, j, k]
end

@kernel function unpermute_y_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    j′ = unpermute_index(j, grid.Ny)
    @inbounds dst[i, j′, k] = src[i, j, k]
end

@kernel function unpermute_z_indices!(dst, src, grid)
    i, j, k = @index(Global, NTuple)
    k′ = unpermute_index(k, grid.Nz)
    @inbounds dst[i, j, k′] = src[i, j, k]
end

permute_kernel! = Dict(
    1 => permute_x_indices!,
    2 => permute_y_indices!,
    3 => permute_z_indices!
)

unpermute_kernel! = Dict(
    1 => unpermute_x_indices!,
    2 => unpermute_y_indices!,
    3 => unpermute_z_indices!
)

permute_indices!(dst, src, arch, grid, dim) =
    launch!(arch, grid, :xyz, permute_kernel![dim], dst, src, grid)

unpermute_indices!(dst, src, arch, grid, dim) =
    launch!(arch, grid, :xyz, unpermute_kernel![dim], dst, src, grid)
