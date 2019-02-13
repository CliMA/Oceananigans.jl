using Oceananigans:
    RegularCartesianGrid,
    Field, CellField, FaceField, FaceFieldX, FaceFieldY, FaceFieldZ, EdgeField,
    VelocityFields, TracerFields, PressureFields, SourceTerms, ForcingFields,
    OperatorTemporaryFields

# Increment and decrement integer a with periodic wrapping. So if n == 10 then
# incmod1(11, n) = 1 and decmod1(0, n) = 10.
@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

# Difference operators.
δx(g, f, i, j, k) = f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k]

@inline δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k])
@inline δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k])
@inline δx!(g::RegularCartesianGrid, f::EdgeField, δxf::FaceField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k])
@inline δx!(g::RegularCartesianGrid, f::FaceField, δxf::EdgeField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k])

@inline δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k])
@inline δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k])
@inline δy!(g::RegularCartesianGrid, f::EdgeField, δyf::FaceField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k])
@inline δy!(g::RegularCartesianGrid, f::FaceField, δyf::EdgeField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k])

@inline function δz!(g::RegularCartesianGrid, f::CellField, δzf::FaceField, i, j, k)
    if k == 1
        @inbounds δzf.data[i, j, k] = 0
    else
        @inbounds δzf.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
    end
end

@inline function δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField, i, j, k)
    if k == g.Nz
        @inbounds δzf.data[i, j, g.Nz] = f.data[i, j, g.Nz]
    else
        @inbounds δzf.data[i, j, k] =  f.data[i, j, k] - f.data[i, j, k+1]
    end
end

@inline function δz!(g::RegularCartesianGrid, f::EdgeField, δzf::FaceField, i, j, k)
    if k == g.Nz
        @inbounds δzf.data[i, j, g.Nz] = f.data[i, j, g.Nz]
    else
        @inbounds δzf.data[i, j, k] =  f.data[i, j, k] - f.data[i, j, k+1]
    end
end

@inline function δz!(g::RegularCartesianGrid, f::FaceField, δzf::EdgeField, i, j, k)
    if k == 1
        @inbounds δzf.data[i, j, k] = 0
    else
        @inbounds δzf.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
    end
end
