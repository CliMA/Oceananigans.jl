#####
##### Nothing happens when your boundary condition is nothing
#####

fill_west_and_east_halo!(c,  ::Nothing, ::Nothing, args...; kwargs...) = NoneEvent()
fill_south_and_north_halo!(c,::Nothing, ::Nothing, args...; kwargs...) = NoneEvent()
fill_bottom_and_top_halo!(c, ::Nothing, ::Nothing, args...; kwargs...) = NoneEvent()

for dir in (:west, :east, :south, :north, :bottom, :top)
        fill_nothing! = Symbol( :fill_, dir, :_halo!)
    alt_fill_nothing! = Symbol(:_fill_, dir, :_halo!)
    @eval begin
        @inline     $fill_nothing!(c, ::Nothing, args...;  kwargs...)         = NoneEvent()
        @inline $alt_fill_nothing!(i, j, grid, c, ::Nothing, args...)         = nothing
        @inline $alt_fill_nothing!(i, j, grid, ::Nothing, ::Nothing, args...) = nothing
        @inline $alt_fill_nothing!(i, j, grid, ::Nothing, args...)            = nothing
    end
end