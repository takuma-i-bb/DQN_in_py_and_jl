using StatsBase: StatsBase
using CUDA
using PyCall
using Flux

struct Transition
    s::Vector
    a::Int
    r::Int
    s_prime::Vector
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
end

function Base.length(replaybuffer::ReplayBuffer)
    length(replaybuffer.buffer)
end

struct DQN
    env::PyObject
    Q_function
end

function DQN(env::PyObject)
    Q_function = Chain(
        Dense(4 => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 2)
    )
    return DQN(env, Q_function)
end

function select_action(dqn::DQN, s::Vector)
    qs = q_forward(dqn, s)
    return argmax(qs)
end

q_forward(dqn::DQN, s::Vector) = dqn.Q_function(s)
loss(x, y) = Flux.Losses.huber_loss(x, y)

function dqn_train!(dqn::DQN, dqn_target::DQN, replaybuffer::ReplayBuffer, optim::Flux.Optimiser)
    data = sample(replaybuffer, batch_size)
end

#main
const α = 0.0005
const γ = 0.98
const buffer_limit = 50000
const batch_size = 32

const Gym = pyimport("gym")
buf = ReplayBuffer()
env = Gym.make("CartPole-v1")
dqn = DQN(env)
dqn_target = deepcopy(dqn)
