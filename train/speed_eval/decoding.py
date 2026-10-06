"""
OpenVLA's action decoding taken apart, so the speed optimisations can replace pieces of it.

Policy.predict (train/infer.py) calls the checkpoint's generate(): one forward over the image
patches and the prompt (the prefill), then one forward per action token with the KV cache, greedy.
Decoder below does exactly that with the model's own submodules, token for token (checked by
speed_eval.py against generate() before every evaluation), in three separable pieces:

    embed(input_ids, pixel_values)  the multimodal input: <s>, 256 projected patches, the prompt
    prefill(embeds)                 the first forward; the KV cache and the logits of the first token
    step(tokens)                    one forward over 1..n tokens on top of the cache

and two ways of producing the 7 tokens:

    greedy        argmax, one token per step: generate(do_sample=False)
    draft-verify  the previous action's tokens are fed as a draft in one step() call; the longest
                  prefix whose tokens equal the argmax of the logits before them is accepted, the
                  first disagreement is replaced by that argmax, and decoding continues from there.
                  This is speculative decoding with the last action as the draft model: greedy
                  decoding's output exactly, in fewer forward passes when the action repeats.

The KV cache is either the model's default (a tuple of tensors, cut back after a rejected draft)
or a StaticCache (fixed buffers, so the step can be torch.compile'd with CUDA graphs; stale
entries past the current position are masked and overwritten, so no cut-back is needed).
"""

import numpy as np
import torch


class Decoder:
    def __init__(self, policy, static_cache=False, max_cache_len=384):
        from transformers.cache_utils import StaticCache

        from peft import PeftModel

        self.policy = policy
        m = policy.model
        # OpenVLAForActionPrediction: under PEFT's wrapper, or on its own once the adapter is merged
        # (transformers gives every model a base_model property, so that name cannot be the test).
        self.vla = m.base_model.model if isinstance(m, PeftModel) else m
        self.lm = self.vla.language_model                                  # LlamaForCausalLM, LoRA layers included
        self.n_tokens = len(policy.names)
        self.device = policy.device
        self.draft = False            # set by the draft-verify optimisation
        self.last_tokens = None
        self.static = static_cache
        self.pos = 0                  # static cache: tokens written so far
        self.forward_calls = 0        # per prediction: prefill + steps, for the log
        if static_cache:
            # Fixed (1, heads, max_cache_len, dim) buffers in every attention layer, in the dtype
            # of the hidden states (float32 for a 4-bit load, whose remaining parameters are
            # upcast; the model's own _setup_cache reads the weight dtype, uint8 there); the model
            # masks the entries past the current position itself.
            for layer in self.lm.model.layers:
                layer.self_attn.past_key_value = StaticCache(self.lm.config, 1, max_cache_len, device=self.device, dtype=policy.input_dtype)
        self.step_fn = self._step    # an optimisation may replace these with compiled versions
        self.embed_patches = lambda px: self.vla.projector(self.vla.vision_backbone(px))

    def close(self):
        """Release the static buffers so the model's own generate() runs as before."""
        if self.static:
            self.lm._reset_cache()

    @torch.inference_mode()
    def warmup(self, prefill_len=281, repeats=3):
        """Run every step shape the decoders use (1 token, and 2..n tokens for a draft) a few times
        on dummy inputs, so a compiled step has recorded all its graphs before anything is timed."""
        hidden = self.lm.config.hidden_size
        dtype = self.policy.input_dtype
        for _ in range(repeats):
            for k in range(1, self.n_tokens + 1):
                _, cache = self.prefill(torch.zeros(1, prefill_len, hidden, device=self.device, dtype=dtype))
                self.step(torch.full((1, k), 31744, device=self.device, dtype=torch.long), cache)
        torch.cuda.synchronize()

    # ---- the pieces --------------------------------------------------------------------------

    def embed(self, input_ids, pixel_values):
        emb = self.vla.get_input_embeddings()(input_ids)
        patches = self.embed_patches(pixel_values)
        return torch.cat([emb[:, :1], patches, emb[:, 1:]], dim=1)

    def prefill(self, embeds):
        """Logits of the first action token and the cache after the prefill."""
        if self.static:
            n = embeds.shape[1]
            out = self.lm(inputs_embeds=embeds, cache_position=torch.arange(n, device=self.device), use_cache=True)
            self.pos = n
            return out.logits[:, -1], None
        out = self.lm(inputs_embeds=embeds, use_cache=True)
        return out.logits[:, -1], out.past_key_values

    def step(self, tokens, cache):
        """Logits at every position of `tokens` (1, k) continued from the cache."""
        k = tokens.shape[1]
        # The cache position goes in as a tensor: a compiled step must not see the Python integer,
        # or it recompiles for every new position.
        cache_position = torch.arange(self.pos, self.pos + k, device=self.device) if self.static else None
        logits, cache = self.step_fn(tokens, cache, cache_position)
        if self.static:
            self.pos += k
        return logits, cache

    def _step(self, tokens, cache, cache_position):
        if self.static:
            out = self.lm(input_ids=tokens, cache_position=cache_position, use_cache=True)
            return out.logits, None
        out = self.lm(input_ids=tokens, past_key_values=cache, use_cache=True)
        return out.logits, out.past_key_values

    def _cut(self, cache, length):
        """The cache with only its first `length` positions (after a rejected draft)."""
        if self.static:
            self.pos = length
            return None
        return tuple((k[:, :, :length], v[:, :, :length]) for k, v in cache)

    # ---- the two decoders ---------------------------------------------------------------------

    @torch.inference_mode()
    def tokens(self, input_ids, pixel_values):
        """The action token ids (n,) for one prompt and image."""
        embeds = self.embed(input_ids, pixel_values)
        logits, cache = self.prefill(embeds)
        self.forward_calls = 1
        base = embeds.shape[1]
        out = [int(logits.argmax(-1))]
        if self.draft and self.last_tokens is not None:
            out = self._verify(out, cache, base)
        else:
            while len(out) < self.n_tokens:
                logits, cache = self.step(torch.tensor([[out[-1]]], device=self.device), cache)
                self.forward_calls += 1
                out.append(int(logits[0, -1].argmax(-1)))
        self.last_tokens = list(out)
        return np.asarray(out)

    def _verify(self, out, cache, base):
        """Greedy decoding with the last action's tokens as the draft, in as few forward passes as it allows.

        Invariant: `out` holds the decided tokens and the cache holds the prefill plus out[:-1] (the
        last decided token has not been fed yet). One pass feeds that token followed by the draft's
        guesses for the positions still open, except the last position, whose token never needs
        feeding. logits[j] is then the greedy token after seq[j]: guesses are confirmed from the
        front while they equal it, the greedy token after the last confirmed one comes for free, and
        the cache is cut back to the decided tokens before the next pass.
        """
        n, draft = self.n_tokens, self.last_tokens
        while len(out) < n:
            seq = ([out[-1]] + draft[len(out):n - 1])[: n - len(out)]
            logits, cache = self.step(torch.tensor([seq], device=self.device), cache)
            self.forward_calls += 1
            argmax = logits[0].argmax(-1).tolist()
            accepted = 0
            while accepted < len(seq) - 1 and argmax[accepted] == seq[accepted + 1]:
                accepted += 1
            out += seq[1:1 + accepted] + [argmax[accepted]]
            cache = self._cut(cache, base + len(out) - 1)
        return out[:n]
