"""CMA-ES training of the World Models controller (Ha & Schmidhuber).

The VAE and MDN-RNN are frozen; CMA-ES evolves the single linear controller
    a_t = W [z_t ; h_t] + b
evaluated in the real CarRacing environment.

Parallelism follows the paper (and Ha's estool): **one whole rollout per worker
process**. A worker is given a parameter vector, runs a *complete* episode locally
(env + VAE + MDN-RNN + controller, all on CPU), and returns a single scalar reward.
Communication is ~900 floats in / 1 float out per rollout -- no per-step IPC -- so it
scales linearly with cores. With popsize P and avg A, each generation submits P*A
independent single-episode tasks, so the avg rollouts parallelise across cores too.

Run as a module (the __main__ guard is required for the 'spawn' start method):
    uv run python train_controller.py --generations 1800 --popsize 64 --avg 16

Modes:
    (default)   real-env training via a persistent multiprocessing Pool
    --dream     roll out inside the MDN-RNN on the GPU (no env); needs a reward head
    --render    watch the saved controller drive in a window (no training)
"""
import os
import time
import argparse
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import cma
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from hydra import initialize_config_dir, compose
from dotenv import load_dotenv
from tqdm import tqdm
import trackio

import model

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    load_dotenv(os.path.join(HERE, ".env"))
    with initialize_config_dir(version_base=None, config_dir=os.path.join(HERE, "conf")):
        overrides = []
        url = os.environ.get("TRACKIO_WRITE_URL")
        if url:
            overrides.append(f'trackio.write_url="{url}"')
        return compose(config_name="config", overrides=overrides)


class Controller(nn.Module):
    """Kept identical to the notebook so saved state_dicts are interchangeable."""

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(
            cfg.controller.state_dim + cfg.controller.hidden_dim,
            cfg.controller.action_dim,
        )

    def forward(self, z, h):
        return self.layer(torch.cat((z, h), dim=-1))


def make_env(max_steps, render_mode="rgb_array"):
    return gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
        max_episode_steps=max_steps,
    )


# --------------------------------------------------------------------------- #
# Single-env rollout, shared by the pool workers and --render (all on one device)
# --------------------------------------------------------------------------- #

def _prep(obs_np, cfg, dev):
    # (96, 96, 3) uint8 -> (1, 3, 64, 64) float in [0,1] on dev
    o = torch.from_numpy(obs_np).permute(2, 0, 1).unsqueeze(0).float()
    o = F.interpolate(o, size=(cfg.dataset.img_size, cfg.dataset.img_size),
                      mode="bilinear", align_corners=False)
    return (o / 255).to(dev)


def _unpack(params, cfg, dev):
    # Layout matches parameters_to_vector -> [weight (out,in) row-major, bias (out,)]
    in_dim = cfg.controller.state_dim + cfg.controller.hidden_dim
    out_dim = cfg.controller.action_dim
    p = torch.as_tensor(params, dtype=torch.float32, device=dev)
    return p[:out_dim * in_dim].view(out_dim, in_dim), p[out_dim * in_dim:]


@torch.no_grad()
def run_episode(Wm, bm, vae, rnn, env, cfg, max_steps):
    """One full episode for one candidate on one env. Returns cumulative reward."""
    dev = Wm.device
    obs = _prep(env.reset()[0], cfg, dev)
    h = torch.zeros(cfg.controller.hidden_dim, device=dev)
    hidden = None
    total = 0.0
    for _ in range(max_steps):
        _, z, _ = vae.encode(obs)                       # mu latent, (1, 32)
        z = z.squeeze(0)                                # (32,)
        a = Wm @ torch.cat([z, h]) + bm                 # (3,)
        a = torch.cat([torch.tanh(a[:1]), torch.sigmoid(a[1:])])  # steer[-1,1], gas/brake[0,1]
        obs_np, r, terminated, truncated, _ = env.step(a.cpu().numpy())
        total += r
        _, _, _, hidden, out = rnn(z.view(1, 1, -1), a.view(1, 1, -1), hidden)
        h = out.reshape(-1)                             # next h_t for the controller
        if terminated or truncated:
            break
        obs = _prep(obs_np, cfg, dev)
    return total


# --------------------------------------------------------------------------- #
# Worker-per-rollout parallelism (the paper's design)
# --------------------------------------------------------------------------- #

_WK = {}  # per-worker persistent state (models + env), set once by the initializer


def _worker_init(cfg, vae_path, rnn_path, max_steps):
    torch.set_num_threads(1)                            # one core per worker; no oversubscription
    vae = model.AutoEncoder(cfg)
    vae.load_state_dict(torch.load(vae_path, map_location="cpu", weights_only=True))
    vae.eval()
    rnn = model.RNN(cfg)
    rnn.load_state_dict(torch.load(rnn_path, map_location="cpu", weights_only=True))
    rnn.eval()
    _WK.update(cfg=cfg, vae=vae, rnn=rnn, env=make_env(max_steps), max_steps=max_steps)


def _worker_task(params):
    # One full episode on CPU; returns its scalar reward.
    Wm, bm = _unpack(params, _WK["cfg"], torch.device("cpu"))
    return run_episode(Wm, bm, _WK["vae"], _WK["rnn"], _WK["env"], _WK["cfg"], _WK["max_steps"])


def evaluate_pool(pool, solutions, avg):
    """Fitness per candidate, averaged over `avg` rollouts, fully parallel.

    One task per (candidate, rollout) so the avg rollouts spread across cores too.
    """
    tasks = [np.asarray(s, dtype=np.float32) for s in solutions for _ in range(avg)]
    res = np.asarray(pool.map(_worker_task, tasks), dtype=np.float64)
    return res.reshape(len(solutions), avg).mean(axis=1)


# --------------------------------------------------------------------------- #
# Dream mode: roll out inside the MDN-RNN, batched on the GPU (no env)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def dream_evaluate(solutions, rnn, reward_head, cfg, max_steps, avg):
    """Fitness by rolling the controller *inside* the MDN-RNN.

    z_{t+1} is sampled from the RNN's mixture; reward is predicted by reward_head(h_t).
    NOTE: the CarRacing MDN-RNN predicts only next-z, not reward, so this is meaningful
    only with a trained reward head (models/reward-<run>.pt). Untrained -> optimises noise.
    """
    N = len(solutions)
    in_dim = cfg.controller.state_dim + cfg.controller.hidden_dim
    out_dim = cfg.controller.action_dim
    params = torch.tensor(np.array(solutions), dtype=torch.float32, device=cfg.device)
    W = params[:, :out_dim * in_dim].view(N, out_dim, in_dim)       # (N, 3, 288)
    b = params[:, out_dim * in_dim:]                                # (N, 3)

    fit = torch.zeros(N, device=cfg.device)
    for _ in range(avg):
        z = torch.randn(N, cfg.rnn.z_dim, device=cfg.device)        # seed from the VAE prior N(0, I)
        h = torch.zeros(N, cfg.controller.hidden_dim, device=cfg.device)
        hidden = None
        for _ in range(max_steps):
            x = torch.cat([z, h], dim=-1).unsqueeze(-1)             # (N, 288, 1)
            a = torch.bmm(W, x).squeeze(-1) + b                     # (N, 3)
            a = torch.cat([torch.tanh(a[:, :1]), torch.sigmoid(a[:, 1:])], dim=-1)
            fit += reward_head(h).squeeze(-1)                       # predicted reward for this step
            pi, mu, sigma, hidden, out = rnn(z.unsqueeze(1), a.unsqueeze(1), hidden)
            z = rnn.mdn.sample(pi.squeeze(1), mu.squeeze(1), sigma.squeeze(1))  # dreamed next latent
            h = out.squeeze(1)
    return (fit / avg).cpu().numpy()


# --------------------------------------------------------------------------- #
# Watch the saved controller drive (single env, on-screen)
# --------------------------------------------------------------------------- #

def watch(run, cfg, max_steps, episodes=5):
    ctrl = Controller(cfg)                              # CPU
    cpath = os.path.join(HERE, f"models/controller-{run}.pt")
    if os.path.exists(cpath):
        ctrl.load_state_dict(torch.load(cpath, map_location="cpu", weights_only=True))
        print(f"watch: loaded {cpath}")
    else:
        print("watch: no saved controller found, using random init")
    Wm, bm = ctrl.layer.weight.detach(), ctrl.layer.bias.detach()

    vae = model.AutoEncoder(cfg)
    vae.load_state_dict(torch.load(os.path.join(HERE, f"models/vae-{run}.pt"), map_location="cpu", weights_only=True))
    vae.eval()
    rnn = model.RNN(cfg)
    rnn.load_state_dict(torch.load(os.path.join(HERE, f"models/rnn-{run}.pt"), map_location="cpu", weights_only=True))
    rnn.eval()

    env = make_env(max_steps, "human")
    for i in range(episodes):
        print(f"episode {i}: reward {run_episode(Wm, bm, vae, rnn, env, cfg, max_steps):.1f}")
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--popsize", type=int, default=None, help="CMA population (default: cma's own)")
    p.add_argument("--sigma", type=float, default=0.3, help="CMA initial step size")
    p.add_argument("--max-steps", type=int, default=1000, help="steps per rollout (paper: 1000)")
    p.add_argument("--avg", type=int, default=16, help="rollouts averaged per candidate (paper: 16)")
    p.add_argument("--workers", type=int, default=None, help="pool size (default: os.cpu_count())")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="watch the saved controller drive; no training")
    p.add_argument("--profile", action="store_true", help="print rollouts/s each generation")
    p.add_argument("--dream", action="store_true",
                   help="train inside the MDN-RNN dream (latent rollouts, no real env; "
                        "needs models/reward-<run>.pt to be meaningful)")
    args = p.parse_args()

    cfg = load_cfg()
    run = cfg.trackio.run_name
    device = cfg.device
    max_steps = args.max_steps

    if args.render:
        watch(run, cfg, max_steps)
        return

    # Dream needs the RNN (+ reward head) resident in the main process on the GPU.
    reward_head = None
    if args.dream:
        rnn = model.RNN(cfg).to(device)
        rnn.load_state_dict(torch.load(os.path.join(HERE, f"models/rnn-{run}.pt"),
                                       weights_only=True, map_location=device))
        rnn.eval()
        reward_head = nn.Linear(cfg.rnn.hidden_size, 1).to(device)
        rpath = os.path.join(HERE, f"models/reward-{run}.pt")
        if os.path.exists(rpath):
            reward_head.load_state_dict(torch.load(rpath, weights_only=True, map_location=device))
            print(f"dream: loaded reward head from {rpath}")
        else:
            print(f"dream: WARNING no {rpath} -- using an UNTRAINED reward head; "
                  "fitness is not meaningful (real-env training is what reproduces the paper)")
        reward_head.eval()

    x0 = parameters_to_vector(Controller(cfg).parameters()).detach().numpy()
    opts = {"seed": args.seed}
    if args.popsize is not None:
        opts["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)
    N = es.popsize

    pool = None
    if not args.dream:
        workers = args.workers or os.cpu_count()
        pool = get_context("spawn").Pool(
            processes=workers, initializer=_worker_init,
            initargs=(cfg, os.path.join(HERE, f"models/vae-{run}.pt"),
                      os.path.join(HERE, f"models/rnn-{run}.pt"), max_steps),
        )
        print(f"real-env  workers={workers}  population={N}  avg={args.avg}  max_steps={max_steps}")
    else:
        print(f"DREAM  population={N}  avg={args.avg}  max_steps={max_steps}  device={device}")

    use_trackio = True
    try:
        trackio.init(
            name=f"controller-{run}", project=cfg.trackio.project, server_url=cfg.trackio.write_url,
            config={"generations": args.generations, "popsize": N, "sigma": args.sigma,
                    "seed": args.seed, "max_steps": max_steps, "avg": args.avg,
                    "workers": args.workers, "dream": args.dream},
        )
    except Exception as e:
        use_trackio = False
        print(f"trackio disabled: {e}")

    best_ever = -np.inf
    pbar = tqdm(range(args.generations), desc="CMA")
    try:
        for gen in pbar:
            if es.stop():
                print("CMA stop:", es.stop())
                break
            solutions = es.ask()
            t0 = time.perf_counter()
            if args.dream:
                fitnesses = dream_evaluate(solutions, rnn, reward_head, cfg, max_steps, args.avg)
            else:
                fitnesses = evaluate_pool(pool, solutions, args.avg)
            dt = time.perf_counter() - t0
            es.tell(solutions, [-f for f in fitnesses])             # CMA minimizes -> negate

            mean, gen_best = float(fitnesses.mean()), float(fitnesses.max())
            if gen_best > best_ever:
                best_ever = gen_best
                ctrl = Controller(cfg)
                vector_to_parameters(torch.tensor(es.result.xbest, dtype=torch.float32), ctrl.parameters())
                torch.save(ctrl.state_dict(), os.path.join(HERE, f"models/controller-{run}.pt"))

            pbar.set_postfix(mean=f"{mean:.1f}", gen_best=f"{gen_best:.1f}", best=f"{best_ever:.1f}")
            if args.profile:
                nroll = len(solutions) * args.avg
                print(f"[profile] {nroll} rollouts in {dt:.1f}s = {nroll/dt:.1f} rollouts/s")
            if use_trackio:
                trackio.log({"reward/mean": mean, "reward/gen_best": gen_best,
                             "reward/best_ever": best_ever, "reward/min": float(fitnesses.min()),
                             "cma/sigma": float(es.sigma)})
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if use_trackio:
            trackio.finish()

    print(f"done. best mean-reward {best_ever:.1f} -> models/controller-{run}.pt")


if __name__ == "__main__":
    main()
