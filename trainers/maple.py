import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.gating_checkpoint import validate_gate_state, load_prompt_checkpoint

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# Schema ids stored in checkpoints; ids 1/2 are the original static scalar/channel gates.
GATE_SCHEMAS = {"scalar": 1, "channel": 2, "cond_channel": 3, "cond_scalar": 4}
CONDITIONAL_GATES = ("cond_channel", "cond_scalar")


class ConditionalGate(nn.Module):
    """g(x) = 1 + 0.5 * tanh(U gelu(D LN(cls))).

    U is zero-initialised, so g == 1 exactly at the first step and the model starts as the
    original MaPLe. Construction does not advance the global RNG (D uses a private generator),
    which keeps data order and every other initialisation paired with the ``none`` baseline.
    """

    def __init__(self, width, rank, out="channel", init_seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.norm = nn.LayerNorm(width, elementwise_affine=False)
            self.down = nn.Linear(width, rank)
            self.up = nn.Linear(rank, width if out == "channel" else 1)
        generator = torch.Generator().manual_seed(1000 + init_seed)
        bound = 1.0 / math.sqrt(width)
        with torch.no_grad():
            self.down.weight.uniform_(-bound, bound, generator=generator)
            self.down.bias.uniform_(-bound, bound, generator=generator)
            self.up.weight.zero_()
            self.up.bias.zero_()

    def forward(self, cls):
        """cls: [B, width] in any dtype -> fp32 gate [B, width] (or [B, 1])."""
        hidden = F.gelu(self.down(self.norm(cls.float())))
        return 1.0 + 0.5 * torch.tanh(self.up(hidden))


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        self.gate_type = getattr(cfg.TRAINER.MAPLE, "GATE_TYPE", "none")
        if self.gate_type not in GATE_SCHEMAS and self.gate_type != "none":
            raise ValueError("TRAINER.MAPLE.GATE_TYPE must be one of none, " + ", ".join(GATE_SCHEMAS))
        if self.gate_type in CONDITIONAL_GATES:
            # Input-conditioned gates: generated per image from the CLS token entering each
            # prompted visual layer, so they cannot be folded into the static projection W.
            rank = int(getattr(cfg.TRAINER.MAPLE, "GATE_RANK", 32))
            out = "channel" if self.gate_type == "cond_channel" else "scalar"
            self.cond_gates = nn.ModuleList([
                ConditionalGate(layer.out_features, rank, out, init_seed=index)
                for index, layer in enumerate(self.compound_prompt_projections)
            ])
            self.register_buffer("_gate_schema", torch.tensor(
                [1, GATE_SCHEMAS[self.gate_type]], dtype=torch.int64))
        # Static, layer-shared scaling. Zero init consumes no random numbers.
        elif self.gate_type != "none":
            self.gate_alphas = nn.ParameterList([
                nn.Parameter(torch.zeros(() if self.gate_type == "scalar" else (layer.out_features,)))
                for layer in self.compound_prompt_projections
            ])
            self.register_buffer("_gate_schema", torch.tensor(
                [1, 1 if self.gate_type == "scalar" else 2], dtype=torch.int64))


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        # Also protects Dassl resume and callers using strict=False.
        validate_gate_state(self, state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            z = layer(self.compound_prompts_text[index])
            if self.gate_type in ("scalar", "channel"):
                # Evaluate tanh in fp32, then preserve the projection/autocast dtype.
                alpha = self.gate_alphas[index]
                gate = (1.0 + 0.5 * torch.tanh(alpha.float())).to(dtype=z.dtype)
                z = z * gate
            if self.gate_type in CONDITIONAL_GATES:
                # The visual block applies the gate after reading the current CLS token.
                visual_deep_prompts.append((z, self.cond_gates[index]))
                continue
            visual_deep_prompts.append(z)
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None, return_features=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if return_features:
            return logits, image_features, text_features
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print("Trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print("Gate type:", cfg.TRAINER.MAPLE.GATE_TYPE,
              "gate parameters:", sum(p.numel() for n, p in self.model.named_parameters()
                                       if "gate_alphas" in n or "cond_gates" in n))

        if cfg.MODEL.INIT_WEIGHTS:
            load_prompt_checkpoint(self.model, load_checkpoint(cfg.MODEL.INIT_WEIGHTS)["state_dict"])

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _consistency_enabled(self):
        m = self.cfg.TRAINER.MAPLE
        return m.KC_TEXT_WEIGHT > 0 or m.KC_IMAGE_WEIGHT > 0 or m.KC_LOGIT_WEIGHT > 0

    def _build_zero_shot_teacher(self):
        """Frozen original CLIP + fixed hand-crafted prompts (same templates as the zero-shot
        reference). Kept on the trainer, not the model, so checkpoints are unchanged."""
        from trainers.coop import load_clip_to_cpu as load_vanilla_clip
        from trainers.zsclip import CUSTOM_TEMPLATES
        cfg = self.cfg
        teacher = load_vanilla_clip(cfg).to(self.device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [template.format(c.replace("_", " ")) for c in self.dm.dataset.classnames]
        with torch.no_grad():
            text = teacher.encode_text(torch.cat([clip.tokenize(p) for p in prompts]).to(self.device))
        self._zs_teacher = teacher
        self._zs_text = (text / text.norm(dim=-1, keepdim=True)).float()
        print(f"Zero-shot consistency teacher ready ({len(prompts)} classes, template {template!r})")

    def _consistency_loss(self, image, label, logits, image_features, text_features):
        m = self.cfg.TRAINER.MAPLE
        if getattr(self, "_zs_teacher", None) is None:
            self._build_zero_shot_teacher()
        with torch.no_grad():
            zs_image = self._zs_teacher.encode_image(image.type(self._zs_teacher.dtype)).float()
            zs_image = zs_image / zs_image.norm(dim=-1, keepdim=True)
            zs_logits = self._zs_teacher.logit_scale.exp().float() * zs_image @ self._zs_text.t()
        if m.KC_REWEIGHT == "teacher_conf":
            # Teacher reliability from training labels only: w_i = p_zs(y_i | x_i). Sample-wise terms
            # are weighted per image; the class-level text term uses the running mean of w.
            with torch.no_grad():
                w = F.softmax(zs_logits, dim=1).gather(1, label[:, None]).squeeze(1)
                self._kc_w_sum = getattr(self, "_kc_w_sum", 0.0) + w.sum().item()
                self._kc_w_count = getattr(self, "_kc_w_count", 0) + w.numel()
                text_scale = self._kc_w_sum / self._kc_w_count
        elif m.KC_REWEIGHT == "none":
            w, text_scale = torch.ones_like(label, dtype=torch.float32), 1.0
        else:
            raise ValueError("TRAINER.MAPLE.KC_REWEIGHT must be none or teacher_conf")
        parts = {}
        if m.KC_TEXT_WEIGHT > 0:
            parts["kc_text"] = m.KC_TEXT_WEIGHT * text_scale * F.l1_loss(text_features.float(), self._zs_text)
        if m.KC_IMAGE_WEIGHT > 0:
            per_image = (image_features.float() - zs_image).abs().mean(dim=1)
            parts["kc_image"] = m.KC_IMAGE_WEIGHT * (w * per_image).mean()
        if m.KC_LOGIT_WEIGHT > 0:
            per_image = F.kl_div(F.log_softmax(logits.float(), dim=1), F.log_softmax(zs_logits, dim=1),
                                 reduction="none", log_target=True).sum(dim=1) / logits.shape[1]
            parts["kc_logit"] = m.KC_LOGIT_WEIGHT * (w * per_image).mean()
        if m.KC_REWEIGHT != "none":
            parts["kc_w"] = w.mean().detach()
        return parts

    def _compute_loss(self, image, label):
        if not self._consistency_enabled():
            return self.model(image, label), {}
        logits, image_features, text_features = self.model(image, label, return_features=True)
        parts = self._consistency_loss(image, label, logits, image_features, text_features)
        ce = F.cross_entropy(logits, label)
        loss = ce + sum(v for k, v in parts.items() if k != "kc_w")
        return loss, {"ce": ce.item(), **{k: float(v) for k, v in parts.items()}}

    def _grad_norm(self):
        """Measure (and optionally clip) the trainable gradient norm; GRAD_CLIP 0 = off."""
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        max_norm = self.cfg.TRAINER.MAPLE.GRAD_CLIP
        if max_norm > 0:
            return float(torch.nn.utils.clip_grad_norm_(params, max_norm))
        with torch.no_grad():
            return float(torch.norm(torch.stack([p.grad.detach().float().norm() for p in params])))

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss, extra = self._compute_loss(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            grad_norm = self._grad_norm()
            scaler.step(optim)
            scaler.update()
        else:
            loss, extra = self._compute_loss(image, label)
            optim.zero_grad()
            loss.backward()
            grad_norm = self._grad_norm()
            optim.step()

        loss_summary = {"loss": loss.item(), "grad_norm": grad_norm, **extra}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            load_prompt_checkpoint(self._models[name], state_dict)
