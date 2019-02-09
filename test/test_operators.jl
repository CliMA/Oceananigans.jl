include("../src/operators/operators_old.jl")

function test_δxc2f(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = CellField(mm, g)
    f.data .= rand(T, size(g))

    δxf1 = δˣc2f(f.data)

    δxf2 = FaceFieldX(mm, g)
    δx!(g, f, δxf2)

    δxf1 ≈ δxf2.data
end

function test_δxf2c(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = FaceFieldX(mm, g)
    f.data .= rand(T, size(g))

    δxf1 = δˣf2c(f.data)

    δxf2 = CellField(mm, g)
    δx!(g, f, δxf2)

    δxf1 ≈ δxf2.data
end

function test_δyc2f(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = CellField(mm, g)
    f.data .= rand(T, size(g))

    δyf1 = δʸc2f(f.data)

    δyf2 = FaceFieldY(mm, g)
    δy!(g, f, δyf2)

    δyf1 ≈ δyf2.data
end

function test_δyf2c(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = FaceFieldY(mm, g)
    f.data .= rand(T, size(g))

    δyf1 = δʸf2c(f.data)

    δyf2 = CellField(mm, g)
    δy!(g, f, δyf2)

    δyf1 ≈ δyf2.data
end

function test_δzc2f(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = CellField(mm, g)
    f.data .= rand(T, size(g))

    δzf1 = δᶻc2f(f.data)

    δzf2 = FaceFieldZ(mm, g)
    δz!(g, f, δzf2)

    δzf1 ≈ δzf2.data
end

function test_δzf2c(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = FaceFieldZ(mm, g)
    f.data .= rand(T, size(g))

    δzf1 = δᶻf2c(f.data)

    δzf2 = CellField(mm, g)
    δz!(g, f, δzf2)

    δzf1 ≈ δzf2.data
end

function test_avgxc2f(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = CellField(mm, g)
    f.data .= rand(T, size(g))

    avgxf1 = avgˣc2f(f.data)

    avgxf2 = FaceFieldX(mm, g)
    avgx!(g, f, avgxf2)

    avgxf1 ≈ avgxf2.data
end

function test_avgxf2c(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = FaceFieldX(mm, g)
    f.data .= rand(T, size(g))

    avgxf1 = avgˣf2c(f.data)

    avgxf2 = CellField(mm, g)
    avgx!(g, f, avgxf2)

    avgxf1 ≈ avgxf2.data
end

function test_avgyc2f(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = CellField(mm, g)
    f.data .= rand(T, size(g))

    avgyf1 = avgʸc2f(f.data)

    avgyf2 = FaceFieldY(mm, g)
    avgy!(g, f, avgyf2)

    avgyf1 ≈ avgyf2.data
end

function test_avgyf2c(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = FaceFieldY(mm, g)
    f.data .= rand(T, size(g))

    avgyf1 = avgʸf2c(f.data)

    avgyf2 = CellField(mm, g)
    avgy!(g, f, avgyf2)

    avgyf1 ≈ avgyf2.data
end

function test_avgzc2f(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = CellField(mm, g)
    f.data .= rand(T, size(g))

    avgzf1 = avgᶻc2f(f.data)

    avgzf2 = FaceFieldZ(mm, g)
    avgz!(g, f, avgzf2)

    avgzf1 ≈ avgzf2.data
end

function test_avgzf2c(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = FaceFieldZ(mm, g)
    f.data .= rand(T, size(g))

    avgzf1 = avgᶻf2c(f.data)

    avgzf2 = CellField(mm, g)
    avgz!(g, f, avgzf2)

    avgzf1 ≈ avgzf2.data
end

function test_divf2c(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    fx = FaceFieldX(mm, g)
    fy = FaceFieldY(mm, g)
    fz = FaceFieldZ(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    fx.data .= rand(T, size(g))
    fy.data .= rand(T, size(g))
    fz.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    div1 = div_f2c(fx.data, fy.data, fz.data)

    div2 = CellField(mm, g)
    div!(g, fx, fy, fz, div2, tmp)

    div1 ≈ div2.data
end

function test_divc2f(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    fx = CellField(mm, g)
    fy = CellField(mm, g)
    fz = CellField(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    fx.data .= rand(T, size(g))
    fy.data .= rand(T, size(g))
    fz.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    div1 = div_c2f(fx.data, fy.data, fz.data)

    div2 = FaceFieldX(mm, g)
    div!(g, fx, fy, fz, div2, tmp)

    div1 ≈ div2.data
end

function test_div_flux(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    U = VelocityFields(mm, g)
    θ = CellField(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    U.u.data .= rand(T, size(g))
    U.v.data .= rand(T, size(g))
    U.w.data .= rand(T, size(g))
    θ.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    div_flux1 = div_flux_f2c(U.u.data, U.v.data, U.w.data, θ.data)

    div_flux2 = CellField(mm, g)
    div_flux!(g, U.u, U.v, U.w, θ, div_flux2, tmp)

    div_flux1 ≈ div_flux2.data
end

function test_u_dot_grad_u(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    U = VelocityFields(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    U.u.data .= rand(T, size(g))
    U.v.data .= rand(T, size(g))
    U.w.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    u∇u1 = ũ∇u(U.u.data, U.v.data, U.w.data)

    u∇u2 = FaceFieldX(mm, g)
    u∇u!(g, U, u∇u2, tmp)

    u∇u1 ≈ u∇u2.data
end

function test_u_dot_grad_v(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    U = VelocityFields(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    U.u.data .= rand(T, size(g))
    U.v.data .= rand(T, size(g))
    U.w.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    u∇v1 = ũ∇v(U.u.data, U.v.data, U.w.data)

    u∇v2 = FaceFieldY(mm, g)
    u∇v!(g, U, u∇v2, tmp)

    u∇v1 ≈ u∇v2.data
end

function test_u_dot_grad_w(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    U = VelocityFields(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    U.u.data .= rand(T, size(g))
    U.v.data .= rand(T, size(g))
    U.w.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    u∇w1 = ũ∇w(U.u.data, U.v.data, U.w.data)

    u∇w2 = FaceFieldZ(mm, g)
    u∇w!(g, U, u∇w2, tmp)

    u∇w1 ≈ u∇w2.data
end

function test_κ∇²(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    tr = TracerFields(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    κh, κv = 4e-2, 4e-2

    tr.T.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    global Δx = g.Δx; global Δy = g.Δy; global Δz = g.Δz
    global κʰ = κh; global κᵛ = κv;
    κ∇²T1 = κ∇²(tr.T.data)

    κ∇²T2 = CellField(mm, g)
    κ∇²!(g, tr.T, κ∇²T2, κh, κv, tmp)

    κ∇²T1 ≈ κ∇²T2.data
end

function test_𝜈∇²u(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    U = VelocityFields(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    𝜈h, 𝜈v = 4e-2, 4e-2

    U.u.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    global Δx = g.Δx; global Δy = g.Δy; global Δz = g.Δz
    global 𝜈ʰ = 𝜈h; global 𝜈ᵛ = 𝜈v;
    𝜈∇²u1 = 𝜈ʰ∇²u(U.u.data)

    𝜈∇²u2 = FaceFieldX(mm, g)
    𝜈∇²u!(g, U.u, 𝜈∇²u2, 𝜈h, 𝜈v, tmp)

    𝜈∇²u1 ≈ 𝜈∇²u2.data
end

function test_𝜈∇²v(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    U = VelocityFields(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    𝜈h, 𝜈v = 4e-2, 4e-2

    U.v.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    global Δx = g.Δx; global Δy = g.Δy; global Δz = g.Δz
    global 𝜈ʰ = 𝜈h; global 𝜈ᵛ = 𝜈v;
    𝜈∇²v1 = 𝜈ʰ∇²v(U.v.data)

    𝜈∇²v2 = FaceFieldY(mm, g)
    𝜈∇²v!(g, U.v, 𝜈∇²v2, 𝜈h, 𝜈v, tmp)

    𝜈∇²v1 ≈ 𝜈∇²v2.data
end

function test_𝜈∇²w(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)

    U = VelocityFields(mm, g)
    tmp = OperatorTemporaryFields(mm, g)

    𝜈h, 𝜈v = 4e-2, 4e-2

    U.w.data .= rand(T, size(g))

    global V = g.V; global Aˣ = g.Ax; global Aʸ = g.Ay; global Aᶻ = g.Az
    global Δx = g.Δx; global Δy = g.Δy; global Δz = g.Δz
    global 𝜈ʰ = 𝜈h; global 𝜈ᵛ = 𝜈v;
    𝜈∇²w1 = 𝜈ᵛ∇²w(U.w.data)

    𝜈∇²w2 = FaceFieldZ(mm, g)
    𝜈∇²w!(g, U.w, 𝜈∇²w2, 𝜈h, 𝜈v, tmp)

    𝜈∇²w1 ≈ 𝜈∇²w2.data
end

function test_∇²_ppn(mm::ModelMetadata, g::Grid)
    T = typeof(g.V)
    f = CellField(mm, g)
    f.data .= rand(T, size(g))

    ∇²f1 = laplacian3d_ppn(f.data)

    ∇²f2 = CellField(mm, g)
    ∇²_ppn!(g, f, ∇²f2)

    ∇²f1 ≈ ∇²f2.data
end
