# Oceananigans.jl Documentation


## Grids
```@docs
RegularCartesianGrid
RegularCartesianGrid(N, L, T)
```

## Fields
```@docs
CellField
FaceFieldX
FaceFieldY
FaceFieldZ
CellField(grid::Grid{T}) where T <: AbstractFloat
FaceFieldX(grid::Grid{T}) where T <: AbstractFloat
FaceFieldY(grid::Grid{T}) where T <: AbstractFloat
FaceFieldZ(grid::Grid{T}) where T <: AbstractFloat
```

## Operators
```@docs
δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField)
δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField)
δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField)
δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField)
δz!(g::RegularCartesianGrid, f::CellField, δzf::FaceField)
δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField)
```
