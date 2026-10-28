"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To submit nanoGPT to Slurm from the repository root, example:
$ sbatch sbatch_nanogpt.sh

To submit GPT-2 to Slurm from the repository root, example:
$ sbatch sbatch_gpt2.sh

To run GPT-2 with DDP on 8 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=8 train.py --config_name=gpt2

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import ast
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

#torch.set_default_dtype(torch.float64)

# -----------------------------------------------------------------------------
# config
#
# Default is the small nanoGPT/Shakespeare run. Use --config_name=gpt2 for
# GPT-2 124M on OpenWebText. GPT-2 in CW reads data/checkpoints from ~/.cache.
config_name = 'nanogpt'  # 'nanogpt' or 'gpt2'
cache_base = os.path.expanduser('~/.cache')

# shared defaults
out_dir = 'out'
data_root = 'data'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
attn_dropout = None  # None -> use `dropout`
mlp_dropout = None   # None -> use `dropout`
resid_dropout = None # None -> use `attn_dropout`
bias = False # do we use bias inside LayerNorm and Linear layers?
normalization_type = "full_bw" # "layer_norm", "full_bw", or "center_only"
batch_center_mode = "running_mean" # "leave_one_out", "running_mean", or "calibrated"
bw_iter_calibration = False
bw_iter_calibration_num_batches = 10
bw_group_size = 0  # 0 -> derive automatically
bw_momentum = 0.1
bw_fix_factor = 0.9
bw_warmup = "diag"  # "diag", "diag_ema", or "none"
bw_inference_ema = False  # leave_one_out only
bw_grad_scale = 0.0  # gradient boost on BW statistics; 0 -> per-mode default
bw_calib_center = "loo"  # calibrated only: "loo", "blend_loo_cov", or "blend" (runs away)
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
backend = 'nccl' # 'nccl', 'gloo', etc.
device = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
) # examples: 'cpu', 'mps', 'cuda', 'cuda:0', 'cuda:1'
dtype = 'float32'  # 'float32', 'bfloat16', 'float16', or 'float64'
compile = False # use PyTorch 2.0 to compile the model to be faster
seed = 1337


def _cli_value(name, default):
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


def apply_nanogpt_config():
    global out_dir, data_root, eval_interval, eval_iters, log_interval
    global always_save_checkpoint, wandb_log, wandb_project, wandb_run_name
    global dataset, gradient_accumulation_steps, batch_size, block_size
    global n_layer, n_head, n_embd, dropout, attn_dropout, mlp_dropout
    global resid_dropout, learning_rate, max_iters, lr_decay_iters, min_lr
    global beta2, warmup_iters

    out_dir = 'out-shakespeare-char'
    data_root = 'data'
    eval_interval = 250
    eval_iters = 200
    log_interval = 10
    always_save_checkpoint = False
    wandb_log = True
    wandb_project = 'shakespeare-char'
    wandb_run_name = 'mini-gpt'
    dataset = 'shakespeare_char'
    gradient_accumulation_steps = 1
    batch_size = 64
    block_size = 256
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.0
    attn_dropout = 0.0  # was 0.2
    mlp_dropout = 0.0  # was 0.2
    resid_dropout = 0.0  # was 0.2
    learning_rate = 1e-3
    max_iters = 5000
    lr_decay_iters = 5000
    min_lr = 1e-4
    beta2 = 0.99
    warmup_iters = 100


def apply_gpt2_config():
    global out_dir, data_root, eval_interval, eval_iters, log_interval
    global always_save_checkpoint, wandb_log, wandb_project, wandb_run_name
    global dataset, gradient_accumulation_steps, batch_size, block_size
    global n_layer, n_head, n_embd, dropout, attn_dropout, mlp_dropout
    global resid_dropout, learning_rate, max_iters, lr_decay_iters, min_lr
    global weight_decay, beta2, warmup_iters

    out_dir = os.path.join(cache_base, 'out')
    data_root = cache_base
    eval_interval = 1000
    eval_iters = 200
    log_interval = 10
    always_save_checkpoint = True
    wandb_log = True
    wandb_project = 'owt'
    wandb_run_name = 'gpt2-124M'
    dataset = 'openwebtext'
    # 12 batch * 1024 block * 5 gradaccum * 8 GPUs = 491,520 tokens.
    # DDP divides this by world size below, so each GPU runs 5 micro-steps.
    gradient_accumulation_steps = 5 * 8
    batch_size = 12
    block_size = 1024
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0
    attn_dropout = None
    mlp_dropout = None
    resid_dropout = None
    learning_rate = 6e-4
    max_iters = 600000
    lr_decay_iters = 600000
    min_lr = 6e-5
    weight_decay = 1e-1
    beta2 = 0.95
    warmup_iters = 2000


config_name = _cli_value("config_name", config_name)
if config_name == 'nanogpt':
    apply_nanogpt_config()
elif config_name == 'gpt2':
    apply_gpt2_config()
else:
    raise ValueError("config_name must be 'nanogpt' or 'gpt2'")

print(f"Using config_name: {config_name}")

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

for arg in sys.argv[1:]:
    if not arg.startswith("--") or "=" not in arg:
        raise ValueError(f"Expected config override in --key=value format, got {arg!r}")
    key, value = arg[2:].split("=", 1)
    if key not in config_keys:
        raise ValueError(f"Unknown config key: {key}")
    try:
        value = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    if not isinstance(value, type(globals()[key])):
        raise TypeError(
            f"Expected {key} to be {type(globals()[key]).__name__}, "
            f"got {type(value).__name__}"
        )
    print(f"Overriding {key} = {value}")
    globals()[key] = value

# Derive calibration from the mode string so experiments use one knob.
bw_iter_calibration = (
    normalization_type != "layer_norm" and batch_center_mode == "calibrated"
)

print(f"normalization_type: {normalization_type}")
print(f"batch_center_mode: {batch_center_mode}")
print(f"bw_iter_calibration: {bw_iter_calibration}")

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

# minimal patch to disable DDP for shakespeare run
if 'out-shakespeare-char' in out_dir:
    ddp = False                # turn off distributed sync
    ddp_world_size = 1         # no scaling
    rank = int(os.environ.get('RANK', 0))          # rank 0..7
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if device.startswith('cuda'):
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    master_process = (rank == 0)   # only rank 0 logs
    seed_offset = 0
else:
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
if device.startswith('cuda'):
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if device.startswith('cuda') else 'mps' if device == 'mps' else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float64': torch.float64}[dtype]
use_autocast = device_type == 'cuda' or (device_type == 'mps' and dtype in ('float16', 'bfloat16'))
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if use_autocast else nullcontext()

# poor man's data loader
data_dir = os.path.join(data_root, dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  attn_dropout=attn_dropout, mlp_dropout=mlp_dropout,
                  resid_dropout=resid_dropout,
                  normalization_type=normalization_type,
                  batch_center_mode=batch_center_mode,
                  bw_group_size=bw_group_size or None, bw_momentum=bw_momentum,
                  bw_fix_factor=bw_fix_factor, bw_warmup=bw_warmup,
                  bw_inference_ema=bw_inference_ema,
                  bw_grad_scale=bw_grad_scale,
                  bw_calib_center=bw_calib_center) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler('cuda', enabled=(device_type == 'cuda' and dtype == 'float16'))
else:
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == 'cuda' and dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)  # DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key, relogin=True, host='https://api.wandb.ai')
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
if hasattr(raw_model, "_orig_mod"):  # torch.compile wrapper
    raw_model = raw_model._orig_mod

@torch.no_grad()
def calibrate_bw_iteration_stats():
    """Calibrate running_mean/running_cov from random batches before each training iteration."""
    if not bw_iter_calibration:
        return
    was_training = model.training
    try:
        # Sample fresh random batches each iteration
        calib_batches = [get_batch('train')[0] for _ in range(bw_iter_calibration_num_batches)]

        raw_model.bw_calibration_reset()
        raw_model.bw_calibration_enable(True)

        # Collect stats in train mode so dropout matches the training forward.
        # @torch.no_grad() keeps calibration from affecting gradients.
        model.train()
        for Xc in calib_batches:
            with ctx:
                model(Xc)
    finally:
        raw_model.bw_calibration_enable(False)
        raw_model.bw_calibration_commit()
        # Disable running stats updates so calibrated stats aren't overwritten during training
        raw_model.set_bw_update_running_stats(False)
        raw_model.set_bw_update_running_cov_stats(False)
        if was_training:
            model.train()

if bw_iter_calibration:
    # Calibrated mode uses freshly collected running stats from iter 0, so avoid
    # the diagonal/full-cov warmup branch and its eval_interval mode switch.
    raw_model.set_bw_cov_warmup(False)
    calibrate_bw_iteration_stats()

@torch.no_grad()
def bw_health():
    """Per-BW-layer health on one training batch, averaged over layers.

    Under no_grad the BW layers whiten with their buffers and leave them
    untouched, so this has no effect on training.
    """
    layers = [l for l in getattr(raw_model, "bw_layers", [])
              if hasattr(l, "running_mean") and hasattr(l, "num_groups")]
    if not layers:
        return None
    rows = []

    def hook(mod, inp, out):
        Xi = inp[0].float()
        B_, T_, C_ = Xi.shape
        ng, gs = mod.num_groups, mod.group_size
        Yg = out.float().view(B_, T_, ng, gs)
        sd = Yg.std(dim=(0, 1)).pow(2).mean().sqrt().clamp_min(1e-12)
        Xg = Xi.view(B_, T_, ng, gs)
        fresh = Xg.mean(dim=(0, 1))
        buf = mod.running_mean.float().view(ng, gs)
        rows.append((
            Xi.pow(2).mean().sqrt().item(),                       # input rms
            sd.item(),                                            # output scale
            (Yg.mean(dim=(0, 1)).norm() / (sd * C_ ** 0.5)).item(),  # resid mean, sd
            ((buf - fresh).norm() / fresh.norm().clamp_min(1e-12)).item(),
        ))

    hs = [l.register_forward_hook(hook) for l in layers]
    try:
        Xh, _ = get_batch('train')
        with ctx:
            model(Xh)
    finally:
        for h in hs:
            h.remove()
    a = np.array(rows)
    return {
        "bw/input_rms": a[:, 0].mean(),
        "bw/input_rms_max": a[:, 0].max(),
        "bw/output_scale": a[:, 1].mean(),
        "bw/resid_mean_sd": a[:, 2].mean(),
        "bw/running_mean_err": a[:, 3].mean(),
    }


running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        health = bw_health()
        if health:
            print(f"step {iter_num}: bw input_rms {health['bw/input_rms']:.4f} "
                  f"(max {health['bw/input_rms_max']:.4f}), "
                  f"output_scale {health['bw/output_scale']:.4f}, "
                  f"resid_mean_sd {health['bw/resid_mean_sd']:.4f}, "
                  f"running_mean_err {health['bw/running_mean_err']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                **(health or {}),
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    if iter_num == eval_interval:
        raw_model.set_bw_cov_warmup(False)

    # Per-iteration BW mean calibration pass (eval/no-grad) over random batches.
    if bw_iter_calibration:
        calibrate_bw_iteration_stats()

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
