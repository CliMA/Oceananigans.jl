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
