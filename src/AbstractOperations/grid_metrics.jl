struct GridMetric{X, Y, Z, A, G, T, M} <: AbstractOperation{X, Y, Z, A, G, T}
          metric :: M
            grid :: G
    architecture :: A

    function GridMetric{X, Y, Z}(metric::M, grid::G) where {X, Y, Z, M, G}
        arch = architecture(grid)
        A = typeof(arch)
        T = eltype(grid)
        return new{X, Y, Z, A, G, T, M}(metric, grid, arch)
    end
end

@inline getindex(gm::GridMetric, i, j, k) = gm.metric(i, j, k, grid)
