@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

function ∇²_ppn!(f, ∇²f, Nx, Ny, Nz, Δx, Δy, Δz)
    for k in 2:(Nz-1), j in 1:Ny, i in 1:Nx
       ∇²f[i, j, k] = (f[incmod1(i, Nx), j, k] - 2*f[i, j, k] + f[decmod1(i, Nx), j, k]) / Δx^2 +
                      (f[i, incmod1(j, Ny), k] - 2*f[i, j, k] + f[i, decmod1(j, Ny), k]) / Δy^2 +
                      (f[i, j, k+1]            - 2*f[i, j, k] + f[i, j, k-1])            / Δz^2
    end
    for j in 1:Ny, i in 1:Nx
        ∇²f[i, j,   1] = (f[i, j, 2] - f[i, j, 1]) / Δz^2 +
                         (f[incmod1(i, Nx), j, 1] - 2*f[i, j, 1] + f[decmod1(i, Nx), j, 1]) / Δx^2 +
                         (f[i, incmod1(j, Ny), 1] - 2*f[i, j, 1] + f[i, decmod1(j, Ny), 1]) / Δy^2
        ∇²f[i, j, end] = (f[i, j, end-1] - f[i, j, end]) / Δz^2 +
                         (f[incmod1(i, Nx), j, end] - 2*f[i, j, end] + f[decmod1(i, Nx), j, end]) / Δx^2 +
                         (f[i, incmod1(j, Ny), end] - 2*f[i, j, end] + f[i, decmod1(j, Ny), end]) / Δy^2
    end
    nothing
end

function δx!(f, δxf, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds δxf[i, j, k] =  f[incmod1(i, Nx), j, k] - f[i, j, k]
    end
    nothing
end

'''
julia> using CuArrays
julia> xc = rand(100, 100, 100); yc = rand(100, 100, 100);
julia> xg = cu(rand(100, 100, 100)); yg = cu(rand(100, 100, 100));

julia> @btime δx!(xc, yc, 100, 100, 100)
  979.906 μs (0 allocations: 0 bytes)

julia> @time δx!(xg, yg, 100, 100, 100)
   19.749605 seconds (15.00 M allocations: 671.387 MiB, 0.24% gc time)
'''

@inline incmod1(i, n) = 1 + (i % n)

'''
julia> @btime δx!(xc, yc, 100, 100, 100)
  3.715 ms (0 allocations: 0 bytes)

julia> @time δx!(xg, yg, 100, 100, 100)
 19.862191 seconds (15.04 M allocations: 673.130 MiB, 0.25% gc time)
'''
