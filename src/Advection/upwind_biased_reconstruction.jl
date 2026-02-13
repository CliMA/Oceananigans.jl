#####
##### Upwind-biased 3rd-order advection scheme
#####

struct UpwindBiased{N, FT, SI, M} <: AbstractUpwindBiasedAdvectionScheme{N, FT, M}
    advecting_velocity_scheme :: SI

    function UpwindBiased{N, FT, M}(advecting_velocity_scheme::SI) where {N, FT, M, SI}
        return new{N, FT, SI, M}(advecting_velocity_scheme)
    end
end

function UpwindBiased(FT::DataType = Float64; order = 3, minimum_buffer_upwind_order = 1)

    mod(order, 2) == 0 && throw(ArgumentError("UpwindBiased reconstruction scheme is defined only for odd orders"))

    N = Int((order + 1) ÷ 2)
    symmetric_order = ifelse(N > 1, order-1, 2)
    advecting_velocity_scheme = Centered(FT; order = symmetric_order)
    minimum_buffer_upwind_order = max(1, min(N, Int(minimum_buffer_upwind_order)))

    return UpwindBiased{N, FT, minimum_buffer_upwind_order}(advecting_velocity_scheme)
end

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased(order=", 2N-1, ")")

function Base.show(io::IO, a::UpwindBiased)
    print(io, summary(a), '\n')
    if minimum_buffer_upwind_order(a) > 1
        print(io, "├── minimum_buffer_upwind_order: ", minimum_buffer_upwind_order(a), '\n')
    end
    print(io, "└── advecting_velocity_scheme: ", summary(a.advecting_velocity_scheme))
end

Adapt.adapt_structure(to, scheme::UpwindBiased{N, FT, SI, M}) where {N, FT, SI, M} =
    UpwindBiased{N, FT, M}(Adapt.adapt(to, scheme.advecting_velocity_scheme))

on_architecture(to, scheme::UpwindBiased{N, FT, SI, M}) where {N, FT, SI, M} =
    UpwindBiased{N, FT, M}(on_architecture(to, scheme.advecting_velocity_scheme))

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)

for (side, dir) in zip((:ᶠᵃᵃ, :ᵃᶠᵃ, :ᵃᵃᶠ), (:x, :y, :z))
    for (F, bool) in zip((:Any, :Callable), (false, true))
        for FT in fully_supported_float_types
            interp = Symbol(:biased_interpolate_, dir, side)
            @eval begin
                @inline $interp(i, j, k, grid, ::UpwindBiased{1, $FT}, red_order::Int, bias, ψ::$F, args...) =
                    @muladd ifelse(red_order==0, $(stencil_reconstruction(FT, 1, :symmetric, dir, bool)),
                                                 ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 1, :right, dir, bool))))

                @inline function $interp(i, j, k, grid, ::UpwindBiased{2, $FT}, red_order::Int, bias, ψ::$F, args...)
                    @muladd ifelse(red_order==0, $(stencil_reconstruction(FT, 1, :symmetric, dir, bool)),
                            ifelse(red_order==1, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                                                 ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 2, :right, dir, bool)))))
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{3, $FT}, red_order::Int, bias, ψ::$F, args...)
                    @muladd ifelse(red_order==0, $(stencil_reconstruction(FT, 1, :symmetric, dir, bool)),
                            ifelse(red_order==1, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                            ifelse(red_order==2, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 2, :right, dir, bool))),
                                                 ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 3, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 3, :right, dir, bool))))))
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{4, $FT}, red_order::Int, bias, ψ::$F, args...)
                    @muladd ifelse(red_order==0, $(stencil_reconstruction(FT, 1, :symmetric, dir, bool)),
                            ifelse(red_order==1, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                            ifelse(red_order==2, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 2, :right, dir, bool))),
                            ifelse(red_order==3, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 3, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 3, :right, dir, bool))),
                                                 ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 4, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 4, :right, dir, bool)))))))
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{5, $FT}, red_order::Int, bias, ψ::$F, args...)
                    @muladd ifelse(red_order==0, $(stencil_reconstruction(FT, 1, :symmetric, dir, bool)),
                            ifelse(red_order==1, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                            ifelse(red_order==2, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 2, :right, dir, bool))),
                            ifelse(red_order==3, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 3, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 3, :right, dir, bool))),
                            ifelse(red_order==4, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 4, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 4, :right, dir, bool))),
                                                 ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 5, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 5, :right, dir, bool))))))))
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{6, $FT}, red_order::Int, bias, ψ::$F, args...)
                    @muladd ifelse(red_order==0, $(stencil_reconstruction(FT, 1, :symmetric, dir, bool)),
                            ifelse(red_order==1, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                            ifelse(red_order==2, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 2, :right, dir, bool))),
                            ifelse(red_order==3, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 3, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 3, :right, dir, bool))),
                            ifelse(red_order==4, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 4, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 4, :right, dir, bool))),
                            ifelse(red_order==5, ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 5, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 5, :right, dir, bool))),
                                                 ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 6, :left,  dir, bool)),
                                                                           $(stencil_reconstruction(FT, 6, :right, dir, bool)))))))))
                end
            end
        end
    end
end
