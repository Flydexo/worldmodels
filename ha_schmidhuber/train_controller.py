"""Parallel CMA-ES training of the World Models controller (Ha & Schmidhuber).

The VAE and MDN-RNN are frozen; CMA-ES evolves the single linear controller
    a_t = W [z_t ; h_t] + b
evaluated in the real CarRacing environment.

Speed comes from two levels of parallelism:
  * env-level: gymnasium AsyncVectorEnv steps the whole CMA population's
    environments in parallel worker processes (Box2D physics is CPU-bound).
  * batch-level: the shared VAE / MDN-RNN and the per-candidate controller
    run as a single batched forward over the population on the GPU (mps).

Run as a module (the __main__ guard is required for the 'spawn' start method
used on macOS):
    uv run python train_controller.py --generations 300 --avg 16
"""
import os
import argparse
from functools import partial

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
    # Top-level + primitive args so functools.partial(make_env, ...) is picklable
    # for AsyncVectorEnv's 'spawn' workers.
    return gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
        max_episode_steps=max_steps,
    )


def preprocess(obs_np, cfg):
    # (N, 96, 96, 3) uint8 -> (N, 3, 64, 64) float in [0,1] on device
    obs = torch.from_numpy(obs_np).permute(0, 3, 1, 2).to(cfg.device).float()
    obs = F.interpolate(
        obs, size=(cfg.dataset.img_size, cfg.dataset.img_size),
        mode="bilinear", align_corners=False,
    )
    return obs / 255


@torch.no_grad()
def _episode(W, b, envs, vae, rnn, cfg, max_steps):
    """One batched episode over all N lanes; returns per-lane cumulative reward."""
    N = W.shape[0]
    obs, _ = envs.reset()
    obs = preprocess(obs, cfg)
    hidden = None                                                    # LSTM (h, c), all lanes
    h = torch.zeros(N, cfg.controller.hidden_dim, device=cfg.device)  # controller input h_t
    total = np.zeros(N, dtype=np.float64)
    active = np.ones(N, dtype=bool)

    for _ in range(max_steps):
        _, z, _ = vae.encode(obs)                                   # (N, 32) use mu: deterministic latent, no sampling noise
        x = torch.cat([z, h], dim=-1).unsqueeze(-1)                 # (N, 288, 1)
        a = torch.bmm(W, x).squeeze(-1) + b                         # (N, 3) per-candidate linear
        # tanh bounds steering to [-1,1]; sigmoid bounds gas/brake to [0,1]
        a = torch.cat([torch.tanh(a[:, :1]), torch.sigmoid(a[:, 1:])], dim=-1)

        obs_np, reward, terminated, truncated, _ = envs.step(a.cpu().numpy().astype(np.float32))
        total += reward * active                                    # freeze reward after a lane finishes

        _, _, _, hidden, out = rnn(z.unsqueeze(1), a.unsqueeze(1), hidden)  # out: (N, 1, 256)
        h = out.squeeze(1)
        obs = preprocess(obs_np, cfg)

        active = active & ~(terminated | truncated)
        if not active.any():
            break

    return total


@torch.no_grad()
def evaluate(solutions, envs, vae, rnn, cfg, max_steps, avg):
    """Fitness per candidate, averaged over `avg` rollouts.

    mu (deterministic latent) removes VAE sampling noise; averaging over rollouts
    removes the CarRacing random-track noise, which is the dominant source. Paper uses 16.
    """
    N = len(solutions)
    in_dim = cfg.controller.state_dim + cfg.controller.hidden_dim
    out_dim = cfg.controller.action_dim
    # Layout matches parameters_to_vector -> [weight (out,in) row-major, bias (out,)]
    params = torch.tensor(np.array(solutions), dtype=torch.float32, device=cfg.device)
    W = params[:, :out_dim * in_dim].view(N, out_dim, in_dim)       # (N, 3, 288)
    b = params[:, out_dim * in_dim:]                                # (N, 3)

    fit = np.zeros(N, dtype=np.float64)
    for _ in range(avg):
        fit += _episode(W, b, envs, vae, rnn, cfg, max_steps)
    return fit / avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--popsize", type=int, default=None, help="CMA population (default: cma's own)")
    p.add_argument("--sigma", type=float, default=0.3, help="CMA initial step size")
    p.add_argument("--max-steps", type=int, default=1000, help="steps per rollout (paper: 1000)")
    p.add_argument("--avg", type=int, default=16, help="rollouts averaged per candidate (paper: 16)")
    p.add_argument("--render", action="store_true",
                   help="show one env in a window (forces SyncVectorEnv, slower)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_cfg()
    run = cfg.trackio.run_name
    device = cfg.device

    vae = model.AutoEncoder(cfg).to(device)
    vae.load_state_dict(torch.load(os.path.join(HERE, f"models/vae-{run}.pt"), weights_only=True, map_location=torch.device(cfg.device)))
    vae.eval()
    rnn = model.RNN(cfg).to(device)
    rnn.load_state_dict(torch.load(os.path.join(HERE, f"models/rnn-{run}.pt"), weights_only=True, map_location=torch.device(cfg.device)))
    rnn.eval()

    x0 = parameters_to_vector(Controller(cfg).parameters()).detach().numpy()
    opts = {"seed": args.seed}
    if args.popsize is not None:
        opts["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)

    N = es.popsize
    max_steps = args.max_steps
    if args.render:
        # One human-rendered lane; Sync runs in-process so the window works on macOS.
        fns = [partial(make_env, max_steps, "human")] + \
              [partial(make_env, max_steps) for _ in range(N - 1)]
        envs = gym.vector.SyncVectorEnv(fns)
    else:
        envs = gym.vector.AsyncVectorEnv([partial(make_env, max_steps) for _ in range(N)])
    print(f"population={N}  max_steps={max_steps}  avg={args.avg}  "
          f"render={args.render}  device={device}")

    use_trackio = True
    try:
        trackio.init(
            name=f"controller-{run}",
            project=cfg.trackio.project,
            server_url=cfg.trackio.write_url,
            config={
                "generations": args.generations,
                "popsize": N,
                "sigma": args.sigma,
                "seed": args.seed,
                "max_steps": max_steps,
                "avg": args.avg,
            },
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
            fitnesses = evaluate(solutions, envs, vae, rnn, cfg, max_steps, args.avg)
            es.tell(solutions, [-f for f in fitnesses])             # CMA minimizes -> negate

            mean, gen_best = float(fitnesses.mean()), float(fitnesses.max())
            if gen_best > best_ever:
                best_ever = gen_best
                ctrl = Controller(cfg)
                vector_to_parameters(
                    torch.tensor(es.result.xbest, dtype=torch.float32), ctrl.parameters()
                )
                torch.save(ctrl.state_dict(), os.path.join(HERE, f"models/controller-{run}.pt"))

            pbar.set_postfix(mean=f"{mean:.1f}", gen_best=f"{gen_best:.1f}", best=f"{best_ever:.1f}")
            if use_trackio:
                trackio.log({
                    "reward/mean": mean,
                    "reward/gen_best": gen_best,
                    "reward/best_ever": best_ever,
                    "reward/min": float(fitnesses.min()),
                    "cma/sigma": float(es.sigma),
                })
    finally:
        envs.close()
        if use_trackio:
            trackio.finish()

    print(f"done. best mean-reward {best_ever:.1f} -> models/controller-{run}.pt")


if __name__ == "__main__":
    main()
