using Oceananigans:
    RegularCartesianGrid

@inline δx_c2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i,   j, k] - f[i-1, j, k]
@inline δx_f2c(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i+1, j, k] - f[i,   j, k]
@inline δx_e2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i+1, j, k] - f[i,   j, k]
@inline δx_f2e(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i,   j, k] - f[i-1, j, k]

@inline δy_c2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j,   k] - f[i, j-1, k]
@inline δy_f2c(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j+1, k] - f[i, j,   k]
@inline δy_e2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j+1, k] - f[i, j,   k]
@inline δy_f2e(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j,   k] - f[i, j-1, k]

@inline δz_c2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j, k-1] - f[i, j,   k]
@inline δz_f2c(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j,   k] - f[i, j, k+1]
@inline δz_e2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j,   k] - f[i, j, k+1]
@inline δz_f2e(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j, k-1] - f[i, j,   k]

@inline avgx_c2f(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i-1,  j, k])
@inline avgx_f2c(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i+1, j, k] + f[i,    j, k])
@inline avgx_f2e(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i-1,  j, k])

@inline avgy_c2f(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i,  j-1, k])
@inline avgy_f2c(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i, j+1, k] + f[i,    j, k])
@inline avgy_f2e(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i,  j-1, k])

@inline avgz_c2f(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i, j, k] + f[i, j, k-1])
@inline avgz_f2c(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i, j, k+1] + f[i, j, k])
@inline avgz_f2e(g::RegularCartesianGrid{T}, f, i, j, k) where T = @inbounds T(0.5) * (f[i, j, k] + f[i, j, k-1])

@inline fv(g::RegularCartesianGrid{T}, v, f, i, j, k) where T = T(0.5) * f * (avgy_f2c(g, v, i-1,  j, k) + avgy_f2c(g, v, i, j, k))
@inline fu(g::RegularCartesianGrid{T}, u, f, i, j, k) where T = T(0.5) * f * (avgx_f2c(g, u, i,  j-1, k) + avgx_f2c(g, u, i, j, k))

@inline div_f2c(g::RegularCartesianGrid, fx, fy, fz, i, j, k) =
    (δx_f2c(g, fx, i, j, k) / g.Δx) + (δy_f2c(g, fy, i, j, k) / g.Δy) + (δz_f2c(g, fz, i, j, k) / g.Δz)

@inline div_c2f(g::RegularCartesianGrid, fx, fy, fz, i, j, k) =
    (δx_c2f(g, fx, i, j, k) / g.Δx) + (δy_c2f(g, fy, i, j, k) / g.Δy) + (δz_c2f(g, fz, i, j, k) / g.Δz)

@inline function δx_f2c_ab̄ˣ(g::RegularCartesianGrid, a, b, i, j, k)
    @inbounds (a[i+1, j, k] * avgx_c2f(g, b, i+1, j, k) -
               a[i,   j, k] * avgx_c2f(g, b, i,   j, k))
end

@inline function δy_f2c_ab̄ʸ(g::RegularCartesianGrid, a, b, i, j, k)
    @inbounds (a[i, j+1, k] * avgy_c2f(g, b, i, j+1, k) -
               a[i,   j, k] * avgy_c2f(g, b, i, j,   k))
end

@inline function δz_f2c_ab̄ᶻ(g::RegularCartesianGrid, a, b, i, j, k)
    @inbounds (a[i, j,   k] * avgz_c2f(g, b, i, j,   k) -
               a[i, j, k+1] * avgz_c2f(g, b, i, j, k+1))
end

@inline div_flux(g::RegularCartesianGrid, u, v, w, Q, i, j, k) =
	(δx_f2c_ab̄ˣ(g, u, Q, i, j, k) / g.Δx) + (δy_f2c_ab̄ʸ(g, v, Q, i, j, k) / g.Δy) + (δz_f2c_ab̄ᶻ(g, w, Q, i, j, k) / g.Δz)

@inline δx_c2f_ūˣūˣ(g::RegularCartesianGrid, u, i, j, k) =
    avgx_f2c(g, u, i, j, k)^2 - avgx_f2c(g, u, i-1, j, k)^2

@inline function δy_e2f_v̄ˣūʸ(g::RegularCartesianGrid, u, v, i, j, k)
    avgx_f2e(g, v, i, j+1, k) * avgy_f2e(g, u, i, j+1, k) -
    avgx_f2e(g, v, i,   j, k) * avgy_f2e(g, u, i,   j, k)
end

@inline function δz_e2f_w̄ˣūᶻ(g::RegularCartesianGrid, u, w, i, j, k)
    @inbounds avgx_f2e(g, w, i, j,   k) * avgz_f2e(g, u, i, j,   k) -
              avgx_f2e(g, w, i, j, k+1) * avgz_f2e(g, u, i, j, k+1)
end

@inline u∇u(g::RegularCartesianGrid, u, v, w, i, j, k) =
    (δx_c2f_ūˣūˣ(g, u, i, j, k) / g.Δx) + (δy_e2f_v̄ˣūʸ(g, u, v, i, j, k) / g.Δy) + (δz_e2f_w̄ˣūᶻ(g, u, w, i, j, k) / g.Δz)

@inline function δx_e2f_ūʸv̄ˣ(g::RegularCartesianGrid, u, v, i, j, k)
    avgy_f2e(g, u, i+1, j, k) * avgx_f2e(g, v, i+1, j, k) -
    avgy_f2e(g, u, i,   j, k) * avgx_f2e(g, v, i,   j, k)
end

@inline δy_c2f_v̄ʸv̄ʸ(g::RegularCartesianGrid, v, i, j, k) =
    avgy_f2c(g, v, i, j, k)^2 - avgy_f2c(g, v, i, j-1, k)^2

@inline function δz_e2f_w̄ʸv̄ᶻ(g::RegularCartesianGrid, v, w, i, j, k)
    @inbounds avgy_f2e(g, w, i, j,   k) * avgz_f2e(g, v, i, j,   k) -
              avgy_f2e(g, w, i, j, k+1) * avgz_f2e(g, v, i, j, k+1)
end

@inline u∇v(g::RegularCartesianGrid, u, v, w, i, j, k) =
    (δx_e2f_ūʸv̄ˣ(g, u, v, i, j, k) / g.Δx) + (δy_c2f_v̄ʸv̄ʸ(g, v, i, j, k) / g.Δy) + (δz_e2f_w̄ʸv̄ᶻ(g, v, w, i, j, k) / g.Δz)

@inline function δx_e2f_ūᶻw̄ˣ(g::RegularCartesianGrid, u, w, i, j, k)
    avgz_f2e(g, u, i+1, j, k) * avgx_f2e(g, w, i+1, j, k) -
    avgz_f2e(g, u, i,   j, k) * avgx_f2e(g, w, i,   j, k)
end

@inline function δy_e2f_v̄ᶻw̄ʸ(g::RegularCartesianGrid, v, w, i, j, k)
    avgz_f2e(g, v, i, j+1, k) * avgy_f2e(g, w, i, j+1, k) -
    avgz_f2e(g, v, i,   j, k) * avgy_f2e(g, w, i,   j, k)
end

@inline δz_c2f_w̄ᶻw̄ᶻ(g::RegularCartesianGrid, w, i, j, k) =
	avgz_f2c(g, w, i, j, k-1)^2 - avgz_f2c(g, w, i, j, k)^2

@inline u∇w(g::RegularCartesianGrid, u, v, w, i, j, k) =
    (δx_e2f_ūᶻw̄ˣ(g, u, w, i, j, k) / g.Δx) + (δy_e2f_v̄ᶻw̄ʸ(g, v, w, i, j, k) / g.Δy) + (δz_c2f_w̄ᶻw̄ᶻ(g, w, i, j, k) / g.Δz)

@inline δx²_c2f2c(g::RegularCartesianGrid, f, i, j, k) = δx_c2f(g, f, i+1, j, k) - δx_c2f(g, f, i, j, k)
@inline δy²_c2f2c(g::RegularCartesianGrid, f, i, j, k) = δy_c2f(g, f, i, j+1, k) - δy_c2f(g, f, i, j, k)
@inline δz²_c2f2c(g::RegularCartesianGrid, f, i, j, k) = δz_c2f(g, f, i, j,   k) - δz_c2f(g, f, i, j, k+1)

@inline κ∇²(g::RegularCartesianGrid, Q, κh, κv, i, j, k) =
    ((κh/g.Δx^2) * δx²_c2f2c(g, Q, i, j, k)) + ((κh/g.Δy^2) * δy²_c2f2c(g, Q, i, j, k)) + ((κv/g.Δz^2) * δz²_c2f2c(g, Q, i, j, k))

@inline δx²_f2c2f(g::RegularCartesianGrid, f, i, j, k) = δx_f2c(g, f, i, j, k) - δx_f2c(g, f, i-1, j, k)
@inline δy²_f2c2f(g::RegularCartesianGrid, f, i, j, k) = δy_f2c(g, f, i, j, k) - δy_f2c(g, f, i, j-1, k)

@inline δx²_f2e2f(g::RegularCartesianGrid, f, i, j, k) = δx_f2e(g, f, i+1, j, k) - δx_f2e(g, f, i, j, k)
@inline δy²_f2e2f(g::RegularCartesianGrid, f, i, j, k) = δy_f2e(g, f, i, j+1, k) - δy_f2e(g, f, i, j, k)

@inline δz²_f2e2f(g::RegularCartesianGrid, f, i, j, k) = δz_f2e(g, f, i, j,   k) - δz_f2e(g, f, i, j, k+1)
@inline δz²_f2c2f(g::RegularCartesianGrid, f, i, j, k) = δz_f2c(g, f, i, j, k-1) - δz_f2c(g, f, i, j, k)

@inline 𝜈∇²u(g::RegularCartesianGrid, u, 𝜈h, 𝜈v, i, j, k) =
    ((𝜈h/g.Δx^2) * δx²_f2c2f(g, u, i, j, k)) + ((𝜈h/g.Δy^2) * δy²_f2e2f(g, u, i, j, k)) + ((𝜈v/g.Δz^2) * δz²_f2e2f(g, u, i, j, k))

@inline 𝜈∇²v(g::RegularCartesianGrid, v, 𝜈h, 𝜈v, i, j, k) =
    ((𝜈h/g.Δx^2) * δx²_f2e2f(g, v, i, j, k)) + ((𝜈h/g.Δy^2) * δy²_f2c2f(g, v, i, j, k)) + ((𝜈v/g.Δz^2) * δz²_f2e2f(g, v, i, j, k))

@inline 𝜈∇²w(g::RegularCartesianGrid, w, 𝜈h, 𝜈v, i, j, k) =
    ((𝜈h/g.Δx^2) * δx²_f2e2f(g, w, i, j, k)) + ((𝜈h/g.Δy^2) * δy²_f2e2f(g, w, i, j, k)) + ((𝜈v/g.Δz^2) * δz²_f2c2f(g, w, i, j, k))

@inline ∇²(g::RegularCartesianGrid, f, i, j, k) =
	(δx²_c2f2c(g, f, i, j, k) / g.Δx^2) + (δy²_c2f2c(g, f, i, j, k) / g.Δy^2) + (δz²_c2f2c(g, f, i, j, k) / g.Δz^2)

@inline ∇h_u(i, j, k, grid, u, v) = δx_f2c(grid, u, i, j, k) / grid.Δx + δy_f2c(grid, v, i, j, k) / grid.Δy
