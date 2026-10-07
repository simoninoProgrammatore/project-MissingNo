"""Short-term memory (GRU): it must carry information across steps, be wiped at a new
episode, and replaying a sequence must give exactly what was computed while playing."""

import sys
from pathlib import Path

import torch
from missingno_agents import CnnActorCritic, Policy, RecurrentActorCritic, build_network

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from ppo import SelfImitationBuffer  # noqa: E402

OBS = (3, 72, 80)


def frames(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, *OBS), generator=g, dtype=torch.uint8)


def test_build_network():
    assert isinstance(build_network(OBS, 7, "none"), CnnActorCritic)
    net = build_network(OBS, 7, "gru", 64)
    assert isinstance(net, RecurrentActorCritic) and net.recurrent and net.memory_size == 64


def test_memory_carries_information_and_is_wiped_at_a_new_episode():
    torch.manual_seed(0)
    net = RecurrentActorCritic(OBS, 7, memory_size=32)
    obs = frames(1)
    _, _, _, v_fresh, _ = net.act(obs, net.initial_state(1))
    _, _, _, _, remembered = net.act(frames(1, seed=1), net.initial_state(1))
    _, _, _, v_after, _ = net.act(obs, remembered)
    assert not torch.allclose(v_fresh, v_after)  # same screen, different past
    _, _, _, v_wiped, _ = net.act(obs, remembered, done=torch.ones(1))
    assert torch.allclose(v_fresh, v_wiped)  # a new episode forgets the past


def test_replaying_a_sequence_matches_playing_it():
    torch.manual_seed(0)
    net = RecurrentActorCritic(OBS, 7, memory_size=32)
    steps, batch = 6, 2
    obs = frames(steps * batch).reshape(steps, batch, *OBS)
    done = torch.zeros(steps, batch)
    done[3, 1] = 1.0  # game 1 starts a new episode at step 3
    state0 = net.initial_state(batch)
    state, logps, values, actions = state0, [], [], []
    with torch.no_grad():
        for t in range(steps):
            state = state * (1 - done[t]).unsqueeze(-1)
            a, logp, _, v, state = net.act(obs[t], state)
            actions.append(a)
            logps.append(logp)
            values.append(v)
        logp_seq, _, value_seq = net.act_sequence(obs, state0, done, torch.stack(actions))
    assert torch.allclose(logp_seq, torch.stack(logps), atol=1e-5)
    assert torch.allclose(value_seq, torch.stack(values), atol=1e-5)


def test_policy_loads_a_recurrent_checkpoint_and_resets(tmp_path):
    net = RecurrentActorCritic(OBS, 7, memory_size=16)
    path = tmp_path / "m.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "obs_shape": OBS,
            "n_actions": 7,
            "config": {"memory": "gru", "memory_size": 16},
        },
        path,
    )
    policy = Policy(str(path))
    action = policy(frames(1)[0].numpy())
    assert 0 <= action < 7
    assert policy.state.abs().sum() > 0
    policy.reset()
    assert policy.state.abs().sum() == 0


def test_self_imitation_keeps_the_memory_of_the_moment():
    buf = SelfImitationBuffer(4, OBS, memory_size=8)
    buf.add(frames(2), torch.tensor([1, 2]), torch.tensor([0.5, 1.0]), torch.ones(2, 8))
    obs, act, ret, mem = buf.sample(3)
    assert mem.shape == (3, 8) and torch.all(mem == 1)
    plain = SelfImitationBuffer(4, OBS)
    plain.add(frames(1), torch.tensor([1]), torch.tensor([0.5]))
    assert plain.sample(1)[3] is None
