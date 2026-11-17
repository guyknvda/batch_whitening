# ns_chol vs cholesky in renorm_fast_whiten given the same cached W_R: same output
# and gradients; the cached next W is symmetric and whitens the batch covariance.
import sys, torch


def cached_W(cache):
    return cache["ns_job"]["W"] if "ns_job" in cache else cache["W"]

from model import renorm_fast_whiten, shrink_cov
torch.manual_seed(0)
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
B, T, C = (4, 64, 32) if dev == "cpu" else (16, 256, 384)
X0 = (torch.randn(B, T, C, device=dev, dtype=torch.float64 if dev == "cpu" else torch.float32) @
      (torch.eye(C, device=dev) + 0.3 * torch.randn(C, C, device=dev)).to(torch.float64 if dev == "cpu" else torch.float32)) + 0.5
dt = X0.dtype
A0 = torch.cov(X0.reshape(-1, C).T) + 0.1 * torch.eye(C, device=dev, dtype=dt)
W0 = torch.linalg.inv(torch.linalg.cholesky(A0))
dY = torch.randn(B, T, C, device=dev, dtype=dt)
res = {}
for wh in ("cholesky", "ns_chol"):
    for pe in (None, 1e-5):
        X = X0.clone().requires_grad_()
        g = (1 + 0.1 * torch.randn(C, device=dev, dtype=dt, generator=torch.Generator(dev).manual_seed(1))).requires_grad_()
        b = torch.zeros(C, device=dev, dtype=dt, requires_grad=True)
        rm, rc = X0.reshape(-1, C).mean(0).clone(), A0.clone()
        cache = {"key": (rc.data_ptr(), rc._version), "W": W0.clone()}
        Y = renorm_fast_whiten(X, g, b, rm, rc, 1e-5, 0.1, 0.9, "running_mean", True, True, cache, "diag",
                               whiten=wh, ns_iters=10, side_stream=dev != "cpu", post_eps=pe)
        (Y * dY).sum().backward()
        if dev != "cpu": torch.cuda.synchronize()
        res[wh, pe] = (Y.detach(), X.grad, g.grad, b.grad, cached_W(cache).reshape(C, C), shrink_cov(rc.reshape(C, C), 0.9, "diag"))
for pe in (None, 1e-5):
    a, c = res["cholesky", pe], res["ns_chol", pe]
    d = [((u - w).abs().max() / u.abs().max()).item() for u, w in zip(a[:4], c[:4])]
    W, A = c[4], c[5]
    I = torch.eye(C, device=dev, dtype=dt)
    print(f"post_eps={pe}: rel diff Y/dX/dg/db", ["%.1e" % x for x in d],
          "| W sym %.1e" % (W - W.T).abs().max().item(), "| WAW-I %.1e" % (W @ A @ W - I).abs().max().item(),
          "| chol W lower-tri", bool(torch.allclose(a[4], torch.tril(a[4]))))

# Several steps on the side stream (NS deferred to the backward) against eager without a side stream.
if dev != "cpu":
    outs = {}
    for ss in (False, True):
        torch.manual_seed(2)
        g = torch.ones(C, device=dev, requires_grad=True); b = torch.zeros(C, device=dev, requires_grad=True)
        rm, rc = torch.zeros(C, device=dev), torch.eye(C, device=dev)
        cache, ys = {}, []
        for it in range(5):
            X = (X0 + 0.1 * it).clone().requires_grad_()
            Y = renorm_fast_whiten(X, g, b, rm, rc, 1e-5, 0.3, 0.8, "running_mean", True, True, cache, "diag",
                                   whiten="ns_chol", ns_iters=10, side_stream=ss, post_eps=1e-5)
            (Y * dY).sum().backward()
            ys.append((Y.detach().clone(), X.grad.clone()))
        torch.cuda.synchronize()
        outs[ss] = ys, cached_W(cache).clone(), "ns_job" in cache
    print("deferred:", outs[True][2], "| 5 steps max rel diff Y, dX:",
          ["%.1e/%.1e" % (((a[0] - c[0]).abs().max() / a[0].abs().max()).item(), ((a[1] - c[1]).abs().max() / a[1].abs().max()).item())
           for a, c in zip(outs[False][0], outs[True][0])],
          "| W %.1e" % ((outs[False][1] - outs[True][1]).abs().max() / outs[False][1].abs().max()).item())
