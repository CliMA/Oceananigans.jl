#####
##### Operators of the form grid_metric * q
#####

for metric in (:Δ, :A), dir in (:x, :y, :z), LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    
    operator    = Symbol(metric, dir, :_q, LX, LY, LZ)
    grid_metric = Symbol(metric, dir, LX, LY, LZ)

    @eval begin
        @inline $operator(i, j, k, grid, q) = @inbounds $grid_metric(i, j, k, grid) * q[i, j, k]
        @inline $operator(i, j, k, grid, f::Function, args...) = $grid_metric(i, j, k, grid) * f(i, j, k, grid, args...)
    end   
end
