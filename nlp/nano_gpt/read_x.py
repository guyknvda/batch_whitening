import torch

tensor_path = "/home/administrator/batch_whitening/nlp/nano_gpt/tensor_5f8160e824ad449185a42308d775abb7.pt"
X = torch.load(tensor_path, weights_only=True)
print("Loaded tensor X:", X)
tensor_type = X.dtype

# Convert to float64 and see if the problem of negative eigenvalues persists
print(f"The type of the loaded tensor is: {tensor_type}, X.shape: {X.shape}")

# ============= float32 ==================

# Move the last dim, C, to be first, and then reshape to 2D tensor, keeping the first dim C intact
# so we simply flatten the last two dims.
Xtmp = X.permute(2, 0, 1)
Xtmp = Xtmp.reshape(Xtmp.shape[0], -1)

mean = Xtmp.mean(dim=1, keepdim=True)
X_centered = Xtmp - mean
X_centered_square = X_centered @ X_centered.T
cov = X_centered_square / Xtmp.size(1)

#cov = torch.cov(Xtmp, correction=0)

# ============= float64 ==================

X_64 = X.to(torch.float64)

Xtmp_64 = X_64.permute(2, 0, 1)
Xtmp_64 = Xtmp_64.reshape(Xtmp.shape[0], -1)

mean_64 = Xtmp_64.mean(dim=1, keepdim=True)
X_centered_64 = Xtmp_64 - mean_64
X_centered_square_64 = X_centered_64 @ X_centered_64.T
cov_64 = X_centered_square_64 / Xtmp_64.size(1)

#cov_64 = torch.cov(Xtmp_64, correction=0)

# ============= Compare ==================

diff = Xtmp_64 - Xtmp.to(torch.float64)
print(f"Xtmp Difference mean: {diff.mean()}, max: {diff.max()}")

diff = mean_64 - mean.to(torch.float64)
print(f"mean Difference mean: {diff.mean()}, max: {diff.max()}")

diff = X_centered_64 - X_centered.to(torch.float64)
print(f"X_centered Difference mean: {diff.mean()}, max: {diff.max()}")
print(f"X_centered min: {X_centered.min()}, max: {X_centered.max()}")
print(f"X_centered_64 min: {X_centered_64.min()}, max: {X_centered_64.max()}")

diff = X_centered_square_64 - X_centered_square.to(torch.float64)
print(f"X_centered_square Difference mean: {diff.mean()}, max: {diff.max()}")
print(f"X_centered_square min: {X_centered_square.min()}, max: {X_centered_square.max()}")
print(f"X_centered_square_64 min: {X_centered_square_64.min()}, max: {X_centered_square_64.max()}")

diff = cov_64 - cov.to(torch.float64)
print(f"cov Difference mean: {diff.mean()}, max: {diff.max()}")

if not (cov == cov.T).all():
    print("cov Not symmetric")
# If using cov.to(torch.float64) then there are no negative eigenvalues. So the overflow is in the eigenvalues computation itself.
eigvals, eigvecs = torch.linalg.eigh(cov)
if not (eigvals >= 0).all():
    print("cov Not PSD")
    print(f'cov.shape: {cov.shape}')
    print(f'Five smallest eigenvalues: {eigvals[:5]}')


if not (cov_64 == cov_64.T).all():
    print("cov_64 Not symmetric")
eigvals, eigvecs = torch.linalg.eigh(cov_64)
if not (eigvals >= 0).all():
    print("cov_64 Not PSD")
    print(f'cov_64.shape: {cov_64.shape}')
    print(f'Five smallest eigenvalues: {eigvals[:5]}')