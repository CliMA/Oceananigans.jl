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

function δx!(f, δxf, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny
        for i in 1:(Nx-1)
            @inbounds δxf[i, j, k] =  f[i+1, j, k] - f[i, j, k]
        end
        @inbounds δxf[end, j, k] =  f[1, j, k] - f[end, j, k]
    end
    nothing
end

'''
julia> using BenchmarkTools
julia> Nx, Ny, Nz = 128, 128, 128;
julia> xc = rand(Nx, Ny, Nz); yc = zeros(Nx, Ny, Nz);
julia> @btime δx!(xc, yc, Nx, Ny, Nz)
  1.409 ms (0 allocations: 0 bytes)
'''

'''
~ nvprof --profile-from-start off /home/alir/julia/bin/julia
julia> using CUDAnative, CUDAdrv, CuArrays
julia> Nx, Ny, Nz = 128, 128, 128;
julia> xg = cu(rand(Nx, Ny, Nz)); yg = cu(rand(Nx, Ny, Nz));
julia> CUDAdrv.@profile @cuda threads=128 δx!(xg, yg, Nx, Ny, Nz)
julia> CUDAdrv.@profile @cuda threads=128 δx!(xg, yg, Nx, Ny, Nz)

==54821== Profiling application: /home/alir/julia/bin/julia
==54821== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  590.15ms         2  295.08ms  294.44ms  295.72ms  ptxcall__x__1

==54821== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 98.85%  9.3003ms         1  9.3003ms  9.3003ms  9.3003ms  cuModuleLoadDataEx
  0.86%  81.203us         2  40.601us  40.423us  40.780us  cuLaunchKernel
  0.08%  7.6330us         1  7.6330us  7.6330us  7.6330us  cuProfilerStart
  0.08%  7.6270us         4  1.9060us     303ns  4.2190us  cuCtxGetCurrent
  0.06%  5.6430us         1  5.6430us  5.6430us  5.6430us  cuDeviceGetCount
  0.04%  3.7120us         4     928ns     196ns  2.3920us  cuDeviceGetAttribute
  0.01%     863ns         1     863ns     863ns     863ns  cuModuleGetFunction
  0.01%     758ns         2     379ns     314ns     444ns  cuDeviceGet
  0.01%     625ns         1     625ns     625ns     625ns  cuCtxGetDevice
'''
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
