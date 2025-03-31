using GLMakie

points = Observable(Point2f[randn(2)])

fig, ax = scatter(points)
limits!(ax, -4, 4, -4, 4)

io = VideoStream(fig, format="mp4", framerate=12, compression=20)

nframes = 120
for i = 1:nframes
    new_point = Point2f(randn(2))
    points[] = push!(points[], new_point)
    recordframe!(io)
end
save("test.webm", io)
