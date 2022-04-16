
#####
##### First derivative operators
#####

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    
    x_derivative = Symbol(:∂x, LX, LY, LZ)
    x_spacing    = Symbol(:Δx, LX, LY, LZ)
    x_difference = Symbol(:δx, LX, :ᵃ, :ᵃ)

    y_derivative = Symbol(:∂y, LX, LY, LZ)
    y_spacing    = Symbol(:Δy, LX, LY, LZ)
    y_difference = Symbol(:δy, :ᵃ, LY, :ᵃ)

    z_derivative = Symbol(:∂z, LX, LY, LZ)
    z_spacing    = Symbol(:Δz, LX, LY, LZ)
    z_difference = Symbol(:δz, :ᵃ, :ᵃ, LZ)

    @eval begin
        @inline $x_derivative(i, j, k, grid, c) = $x_difference(i, j, k, grid, c) / $x_spacing(i, j, k, grid)
        @inline $y_derivative(i, j, k, grid, c) = $y_difference(i, j, k, grid, c) / $y_spacing(i, j, k, grid)
        @inline $z_derivative(i, j, k, grid, c) = $z_difference(i, j, k, grid, c) / $z_spacing(i, j, k, grid)
        
        @inline $x_derivative(i, j, k, grid, f::Function, args...) = $x_difference(i, j, k, grid, f, args...) / $x_spacing(i, j, k, grid)
        @inline $y_derivative(i, j, k, grid, f::Function, args...) = $y_difference(i, j, k, grid, f, args...) / $y_spacing(i, j, k, grid)
        @inline $z_derivative(i, j, k, grid, f::Function, args...) = $z_difference(i, j, k, grid, f, args...) / $z_spacing(i, j, k, grid)
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


for dir in (:x, :y, :z), L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ)

     first_order_face = Symbol(:∂,  dir, insert_symbol(dir, :ᶠ, L1, L2)...)
    second_order_face = Symbol(:∂², dir, insert_symbol(dir, :ᶠ, L1, L2)...)
     third_order_face = Symbol(:∂³, dir, insert_symbol(dir, :ᶠ, L1, L2)...)
    fourth_order_face = Symbol(:∂⁴, dir, insert_symbol(dir, :ᶠ, L1, L2)...)

     first_order_center = Symbol(:∂,  dir, insert_symbol(dir, :ᶜ, L1, L2)...)
    second_order_center = Symbol(:∂², dir, insert_symbol(dir, :ᶜ, L1, L2)...)
     third_order_center = Symbol(:∂³, dir, insert_symbol(dir, :ᶜ, L1, L2)...)
    fourth_order_center = Symbol(:∂⁴, dir, insert_symbol(dir, :ᶜ, L1, L2)...)

    @eval begin
        $second_order_face(i, j, k, grid, c) =  $first_order_face(i, j, k, grid, $first_order_center,  c)
         $third_order_face(i, j, k, grid, c) =  $first_order_face(i, j, k, grid, $second_order_center, c)
        $fourth_order_face(i, j, k, grid, c) = $second_order_face(i, j, k, grid, $second_order_face,   c)

        $second_order_center(i, j, k, grid, c) =  $first_order_center(i, j, k, grid, $first_order_face,    c)
         $third_order_center(i, j, k, grid, c) =  $first_order_center(i, j, k, grid, $second_order_face,   c)
        $fourth_order_center(i, j, k, grid, c) = $second_order_center(i, j, k, grid, $second_order_center, c)

        $second_order_face(i, j, k, grid, f::Function, args...) =  $first_order_face(i, j, k, grid, $first_order_center,  f::Function, args...)
         $third_order_face(i, j, k, grid, f::Function, args...) =  $first_order_face(i, j, k, grid, $second_order_center, f::Function, args...)
        $fourth_order_face(i, j, k, grid, f::Function, args...) = $second_order_face(i, j, k, grid, $second_order_face,   f::Function, args...)

        $second_order_center(i, j, k, grid, f::Function, args...) =  $first_order_center(i, j, k, grid, $first_order_face,    f::Function, args...)
         $third_order_center(i, j, k, grid, f::Function, args...) =  $first_order_center(i, j, k, grid, $second_order_face,   f::Function, args...)
        $fourth_order_center(i, j, k, grid, f::Function, args...) = $second_order_center(i, j, k, grid, $second_order_center, f::Function, args...)
    end
end

#####
##### Operators of the form A*∂(q) where A is an area and q is some quantity.
#####

for dir in (:x, :y, :z), LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    
    operator   = Symbol(:A, dir, :_∂, dir, LX, LY, LZ)
    area       = Symbol(:A, dir, LX, LY, LZ)
    derivative = Symbol(:∂, dir, LX, LY, LZ)

    @eval begin
        $operator(i, j, k, grid, c) = $area(i, j, k, grid) * $derivative(i, j, k, grid, c)
    end
end
