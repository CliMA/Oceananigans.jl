#####
##### Nothing happens when your boundary condition is nothing
#####

for dir in (:west, :east, :south, :north, :bottom, :top)
    alt_fill_nothing! = Symbol(:_fill_, dir, :_halo!)
    @eval begin
        @inline     $fill_nothing!(c, ::Nothing, args...;  kwargs...)         = nothing
        @inline $alt_fill_nothing!(i, j, grid, c, ::Nothing, args...)         = nothing
        @inline $alt_fill_nothing!(i, j, grid, ::Nothing, ::Nothing, args...) = nothing
        @inline $alt_fill_nothing!(i, j, grid, ::Nothing, args...)            = nothing
    end
end
