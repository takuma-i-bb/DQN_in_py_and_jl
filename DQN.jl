using StatsBase: StatsBase
using CUDA

struct Transition
    s::Tuple
    a::Int
    r::Int
    s_prime::Tuple
    done::Int
end

struct ReplayBuffer
    buffer::Vector{Transition}

    ReplayBuffer() = new([])
end

function Base.push!(replaybuffer::ReplayBuffer, detum::Transition)
    return push!(replaybuffer.buffer, detum)
end

function sample(replaybuffer::ReplayBuffer, n::Int)
    mini_batch = StatsBase.sample(replaybuffer.buffer, n, replace=false)
    s_list, a_list, r_list, s_prime, done_list = [], [], [], [], []

    for detum in mini_batch
        push!(s_list, detum.s)
        push!(a_list, )
end

function Base.length(replaybuffer::ReplayBuffer)
    length(replaybuffer.buffer)
end

Q_function = Chain(
    Dense(4 => 128, relu),
    Dense(128 => 128, relu),
    Dense(128 => 2)
) |> gpu

loss(x, y) = Flux.Losses.huber_loss(Q_function(x), y)

a = ReplayBuffer()
println(Q_function(rand(4)))
