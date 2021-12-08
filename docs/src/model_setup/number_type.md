# Number type

Passing `float_type=Float64` or `float_type=Float32` to the `Model` constructor causes the model to store all numbers
with 64-bit or 32-bit floating point precision.

!!! note "Avoiding mixed-precision operations"
    When not using `Float64` be careful to not mix different precisions as it could introduce implicit type conversions
    which can negatively effect performance. You can pass the number type desires to many constructors to enforce
    the type you want: e.g. `RectilinearGrid(CPU(), Float32; size=(16, 16, 16), extent=(1, 1, 1))` and
    `IsotropicDiffusivity(Float16; κ=1//7, ν=2//7)`.

!!! warning "Effect of floating point precision on simulation accuracy"
    While we run many tests with both `Float32` and `Float64` it is not clear whether `Float32` is precise enough to
    provide similar accuracy in all use cases. If accuracy is a concern, stick to `Float64`.

    We will be actively investigating the possibility of using lower precision floating point numbers such as `Float32`
    and `Float16` for fluid dynamics as well as the use of alternative number types such as Posits and Sonums.
