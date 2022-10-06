using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Operators: volume
using Oceananigans.Grids: AbstractGrid

@inline get_node(::Bounded, i, N) = min(max(i, 1), N)
@inline get_node(::Periodic, i, N) = ifelse(i < 1, N - i, ifelse(i > N, 1 + i, i))

@inline function get_nearest_nodes(x, y, z, grid, loc)
    i, j, k = fractional_indices(x, y, z, loc, grid)

    # Convert fractional indices to unit cell coordinates 0 <= (ξ, η, ζ) <=1
    # and integer indices (with 0-based indexing).
    ξ, i = modf(i)
    η, j = modf(j)
    ζ, k = modf(k)

    if (ξ, η, ζ) == (0, 0, 0) #particle on grid point special case
        return ((Int(i+1), Int(j+1), Int(k+1), 1), ), 1.0
    else
        nodes = repeat([(1, 1, 1, NaN)], 8)
        _normfactor = 0.0
        @unroll for n=1:8
            # Move around cube corners getting node indices (0 or 1) and distances to them
            # Distance is d when the index is 0, or 1-d when it is 1
            a = 0^(1+(-1)^n)
            di = 0^abs(1-a)+ξ*(-1)^a

            b = 0^(1+(-1)^floor(n/2))
            dj = 0^abs(1-b)+η*(-1)^b

            c = 0^(1+(-1)^floor(n/4))
            dk = 0^abs(1-c)+ζ*(-1)^c

            nodes[n] = (Int(i+1)+a, 
                            Int(j+1)+b, 
                            Int(k+1)+c, 
                            sqrt(di^2+dj^2+dk^2))
            _normfactor += 1 ./sqrt(di^2+dj^2+dk^2)
        end
        return nodes, 1/_normfactor
    end
end

@kernel function calculate_particle_tendency_kernel!(property, tendency, particles, grid::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    p = @index(Global)
    #density = :density in keys(particles.parameters) ? particles.parameters.density : 1
    density = 1

    LX, LY, LZ = location(tendency)
    nodes, normfactor = @inbounds get_nearest_nodes(particles.properties.x[p], particles.properties.y[p], particles.properties.z[p], grid, (LX(), LY(), LZ()))

    @unroll for (i, j, k, d) in nodes 
        # Reflect back on Bounded boundaries or wrap around for Periodic boundaries
        i, j, k = (get_node(TX(), i, grid.Nx), get_node(TY(), j, grid.Ny), get_node(TZ(), k, grid.Nz))

        node_volume = volume(i, j, k, grid, LX(), LY(), LZ())
        value = density * @inbounds property[p] * normfactor / (d * node_volume)
        @inbounds tendency[i, j, k] += value	
    end
end