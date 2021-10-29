const thirty_days = 30days
const Nmonths = 12

#@inline current_time_index(time, interval=2592000, length=12) = mod(unsafe_trunc(Int32, time / interval),     length) + 1
#@inline next_time_index(time, interval=2592000, length=12)    = mod(unsafe_trunc(Int32, time / interval) + 1, length) + 1
#@inline cyclic_interpolate(u₁::Number, u₂, time, interval=2592000) = u₁ + mod(time / interval, 1) * (u₂ - u₁)

@inline current_time_index(time) = mod(unsafe_trunc(Int32, time / 2592000), 12) + 1
@inline next_time_index(time) = mod(unsafe_trunc(Int32, time / 2592000) + 1, 12) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / 2592000, 1) * (u₂ - u₁)

@inline function cyclic_interpolate(τ::AbstractArray, time, interval=2592000, length=12)
    n₁ = current_time_index(time, interval, length)
    n₂ = next_time_index(time, interval, length)
    return cyclic_interpolate.(view(τ, :, :, n₁), view(τ, :, :, n₂), time, interval)
end

