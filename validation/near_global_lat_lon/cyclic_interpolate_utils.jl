thirty_days = 30days
Nmonths = 12

@inline current_time_index(time, interval=thirty_days, length=Nmonths) = mod(trunc(Int, time / interval),     length) + 1
@inline next_time_index(time, interval=thirty_days, length=Nmonths)    = mod(trunc(Int, time / interval) + 1, length) + 1

@inline cyclic_interpolate(u₁, u₂, time, interval=thirty_days) = u₁ + mod(time / interval, 1) * (u₂ - u₁)

@inline function cyclic_interpolate(τ::AbstractArray, time, interval=thirty_days, length=Nmonths)
    n₁ = current_time_index(time, interval, length)
    n₂ = next_time_index(time, interval, length)
    return cyclic_interpolate.(view(τ, :, :, n₁), view(τ, :, :, n₂), time, interval)
end

