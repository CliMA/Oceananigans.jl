# Operators
## Difference operators
```@docs
δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField)
δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField)
δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField)
δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField)
δz!(g::RegularCartesianGrid, f::CellField, δzf::FaceField)
δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField)
```

## Averaging operators
```@docs
avgx!(g::RegularCartesianGrid, f::CellField, favgx::FaceField)
```

## Divergence operators
Building on top of the differencing operators we can define operators that
compute the divergence
```math
\nabla\cdotp\mathbf{f} = \frac{1}{V} \left[ \delta_x \left( A_x f_x \right)
+ \delta_y\left( A_y f_y \right) + \delta_z\left( A_z f_z \right)\right]
```

```@docs
div!(g::RegularCartesianGrid, fx::FaceFieldX, fy::FaceFieldY, fz::FaceFieldZ, div::CellField, tmp::OperatorTemporaryFields)
```
