include("../src/operators/operators_old.jl")

function test_δxc2f(g::Grid)
    T = typeof(g.V)
    f = CellField(g)
    f.data .= rand(T, size(g))

    δxf1 = δˣc2f(f.data)

    δxf2 = FaceFieldX(g)
    δx!(g, f, δxf2)

    δxf1 ≈ δxf2.data
end

function test_δxf2c(g::Grid)
    T = typeof(g.V)
    f = FaceFieldX(g)
    f.data .= rand(T, size(g))

    δxf1 = δˣf2c(f.data)

    δxf2 = CellField(g)
    δx!(g, f, δxf2)

    δxf1 ≈ δxf2.data
end

function test_δyc2f(g::Grid)
    T = typeof(g.V)
    f = CellField(g)
    f.data .= rand(T, size(g))

    δyf1 = δʸc2f(f.data)

    δyf2 = FaceFieldY(g)
    δy!(g, f, δyf2)

    δyf1 ≈ δyf2.data
end

function test_δyf2c(g::Grid)
    T = typeof(g.V)
    f = FaceFieldY(g)
    f.data .= rand(T, size(g))

    δyf1 = δʸf2c(f.data)

    δyf2 = CellField(g)
    δy!(g, f, δyf2)

    δyf1 ≈ δyf2.data
end

function test_δzc2f(g::Grid)
    T = typeof(g.V)
    f = CellField(g)
    f.data .= rand(T, size(g))

    δzf1 = δᶻc2f(f.data)

    δzf2 = FaceFieldZ(g)
    δz!(g, f, δzf2)

    δzf1 ≈ δzf2.data
end

function test_δzf2c(g::Grid)
    T = typeof(g.V)
    f = FaceFieldZ(g)
    f.data .= rand(T, size(g))

    δzf1 = δᶻf2c(f.data)

    δzf2 = CellField(g)
    δz!(g, f, δzf2)

    δzf1 ≈ δzf2.data
end

function test_avgxc2f(g::Grid)
    T = typeof(g.V)
    f = CellField(g)
    f.data .= rand(T, size(g))

    avgxf1 = avgˣc2f(f.data)

    avgxf2 = FaceFieldX(g)
    avgx!(g, f, avgxf2)

    avgxf1 ≈ avgxf2.data
end

function test_avgxf2c(g::Grid)
    T = typeof(g.V)
    f = FaceFieldX(g)
    f.data .= rand(T, size(g))

    avgxf1 = avgˣf2c(f.data)

    avgxf2 = CellField(g)
    avgx!(g, f, avgxf2)

    avgxf1 ≈ avgxf2.data
end

function test_avgyc2f(g::Grid)
    T = typeof(g.V)
    f = CellField(g)
    f.data .= rand(T, size(g))

    avgyf1 = avgʸc2f(f.data)

    avgyf2 = FaceFieldY(g)
    avgy!(g, f, avgyf2)

    avgyf1 ≈ avgyf2.data
end

function test_avgyf2c(g::Grid)
    T = typeof(g.V)
    f = FaceFieldY(g)
    f.data .= rand(T, size(g))

    avgyf1 = avgʸf2c(f.data)

    avgyf2 = CellField(g)
    avgy!(g, f, avgyf2)

    avgyf1 ≈ avgyf2.data
end

function test_avgzc2f(g::Grid)
    T = typeof(g.V)
    f = CellField(g)
    f.data .= rand(T, size(g))

    avgzf1 = avgᶻc2f(f.data)

    avgzf2 = FaceFieldZ(g)
    avgz!(g, f, avgzf2)

    avgzf1 ≈ avgzf2.data
end

function test_avgzf2c(g::Grid)
    T = typeof(g.V)
    f = FaceFieldZ(g)
    f.data .= rand(T, size(g))

    avgzf1 = avgᶻf2c(f.data)

    avgzf2 = CellField(g)
    avgz!(g, f, avgzf2)

    avgzf1 ≈ avgzf2.data
end
