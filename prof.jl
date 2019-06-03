using CUDAdrv, Oceananigans

model = Model(N=(256, 256, 256), L=(100, 100, 100), arch=GPU())
time_step!(model, 1, 1)

CUDAdrv.@profile begin
    time_step!(model, 10, 10)
end

exit()
