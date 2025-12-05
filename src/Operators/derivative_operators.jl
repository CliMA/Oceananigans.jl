#####
##### First derivative operators
#####

for ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ, :ᵃ), ℓ3 in (:ᶜ, :ᶠ, :ᵃ)

    x_derivative  = Symbol(:∂x,   ℓ1, ℓ2, ℓ3)
    rcp_x_spacing = Symbol(:Δx⁻¹, ℓ1, ℓ2, ℓ3)
    x_difference  = Symbol(:δx,   ℓ1, ℓ2, ℓ3)

    y_derivative  = Symbol(:∂y,   ℓ2, ℓ1, ℓ3)
    rcp_y_spacing = Symbol(:Δy⁻¹, ℓ2, ℓ1, ℓ3)
    y_difference  = Symbol(:δy,   ℓ2, ℓ1, ℓ3)

    z_derivative  = Symbol(:∂z,   ℓ2, ℓ3, ℓ1)
    rcp_z_spacing = Symbol(:Δz⁻¹, ℓ2, ℓ3, ℓ1)
    z_difference  = Symbol(:δz,   ℓ2, ℓ3, ℓ1)

    @eval begin
        @inline $x_derivative(i, j, k, grid, c) = $x_difference(i, j, k, grid, c) * $rcp_x_spacing(i, j, k, grid)
        @inline $y_derivative(i, j, k, grid, c) = $y_difference(i, j, k, grid, c) * $rcp_y_spacing(i, j, k, grid)
        @inline $z_derivative(i, j, k, grid, c) = $z_difference(i, j, k, grid, c) * $rcp_z_spacing(i, j, k, grid)

        @inline $x_derivative(i, j, k, grid, c::Number) = zero(grid)
        @inline $y_derivative(i, j, k, grid, c::Number) = zero(grid)
        @inline $z_derivative(i, j, k, grid, c::Number) = zero(grid)

        @inline $x_derivative(i, j, k, grid, f::Function, args...) = $x_difference(i, j, k, grid, f, args...) * $rcp_x_spacing(i, j, k, grid)
        @inline $y_derivative(i, j, k, grid, f::Function, args...) = $y_difference(i, j, k, grid, f, args...) * $rcp_y_spacing(i, j, k, grid)
        @inline $z_derivative(i, j, k, grid, f::Function, args...) = $z_difference(i, j, k, grid, f, args...) * $rcp_z_spacing(i, j, k, grid)

        export $x_derivative
        export $y_derivative
        export $z_derivative
    end
end

#####
##### Second, Third, and Fourth derivatives
#####

@inline insert_symbol(dir, L, L1, L2) =
                      dir == :x ?
                      (L, L1, L2) :
                      dir == :y ?
                      (L1, L, L2) :
                      (L1, L2, L)


for dir in (:x, :y, :z), L1 in (:ᶜ, :ᶠ, :ᵃ), L2 in (:ᶜ, :ᶠ, :ᵃ)

     first_order_face = Symbol(:∂,  dir, insert_symbol(dir, :ᶠ, L1, L2)...)
    second_order_face = Symbol(:∂², dir, insert_symbol(dir, :ᶠ, L1, L2)...)
     third_order_face = Symbol(:∂³, dir, insert_symbol(dir, :ᶠ, L1, L2)...)
    fourth_order_face = Symbol(:∂⁴, dir, insert_symbol(dir, :ᶠ, L1, L2)...)

     first_order_center = Symbol(:∂,  dir, insert_symbol(dir, :ᶜ, L1, L2)...)
    second_order_center = Symbol(:∂², dir, insert_symbol(dir, :ᶜ, L1, L2)...)
     third_order_center = Symbol(:∂³, dir, insert_symbol(dir, :ᶜ, L1, L2)...)
    fourth_order_center = Symbol(:∂⁴, dir, insert_symbol(dir, :ᶜ, L1, L2)...)

    @eval begin
        @inline $second_order_face(i, j, k, grid, c) =  $first_order_face(i, j, k, grid, $first_order_center,  c)
        @inline  $third_order_face(i, j, k, grid, c) =  $first_order_face(i, j, k, grid, $second_order_center, c)
        @inline $fourth_order_face(i, j, k, grid, c) = $second_order_face(i, j, k, grid, $second_order_face,   c)

        @inline $second_order_center(i, j, k, grid, c) =  $first_order_center(i, j, k, grid, $first_order_face,    c)
        @inline  $third_order_center(i, j, k, grid, c) =  $first_order_center(i, j, k, grid, $second_order_face,   c)
        @inline $fourth_order_center(i, j, k, grid, c) = $second_order_center(i, j, k, grid, $second_order_center, c)

        @inline $second_order_face(i, j, k, grid, f::Function, args...) =  $first_order_face(i, j, k, grid, $first_order_center,  f::Function, args...)
        @inline  $third_order_face(i, j, k, grid, f::Function, args...) =  $first_order_face(i, j, k, grid, $second_order_center, f::Function, args...)
        @inline $fourth_order_face(i, j, k, grid, f::Function, args...) = $second_order_face(i, j, k, grid, $second_order_face,   f::Function, args...)

        @inline $second_order_center(i, j, k, grid, f::Function, args...) =  $first_order_center(i, j, k, grid, $first_order_face,    f::Function, args...)
        @inline  $third_order_center(i, j, k, grid, f::Function, args...) =  $first_order_center(i, j, k, grid, $second_order_face,   f::Function, args...)
        @inline $fourth_order_center(i, j, k, grid, f::Function, args...) = $second_order_center(i, j, k, grid, $second_order_center, f::Function, args...)
    end
end

#####
##### Operators of the form A*∂(q) where A is an area and q is some quantity.
#####

for dir in (:x, :y, :z), LX in (:ᶜ, :ᶠ, :ᵃ), LY in (:ᶜ, :ᶠ, :ᵃ), LZ in (:ᶜ, :ᶠ, :ᵃ)

    operator   = Symbol(:A, dir, :_∂, dir, LX, LY, LZ)
    area       = Symbol(:A, dir, LX, LY, LZ)
    derivative = Symbol(:∂, dir, LX, LY, LZ)

    @eval begin
        @inline $operator(i, j, k, grid, c) = $area(i, j, k, grid) * $derivative(i, j, k, grid, c)
    end
end
