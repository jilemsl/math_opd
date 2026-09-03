"""`DistillationTrainer` + a token mask. That mask is the entire contribution.

The base trainer already reduces its loss over

    loss_mask = completion_mask * tool_mask        (distillation_trainer.py)

so an arm is installed by writing the arm's mask into `tool_mask` and calling
the base implementation. Nothing about the JSD, the chunked lm_head projection,
or the grad-accum normalization is duplicated or re-derived here.
"""

import torch

from trl import DistillationConfig, DistillationTrainer

from .mask_utils import ARMS, LEXICAL_ARMS, SCORED_ARMS, baseline_batch_mask, lexical_batch_mask, retention


#: Positions per `lm_head` call in the scoring pass, mirroring the trainer's
#: chunked loss so peak memory does not scale with sequence length.
_SCORE_CHUNK = 512


class MaskedDistillationConfig(DistillationConfig):
    """`DistillationConfig` plus the arm selector.

    `arm` is one of `vanilla`, `random`, `entropy`, `kl`, `v0`, `v1`, `v0ms`,
    `v2`. `budget_variant` names the mask whose retention the baselines must
    match -- v0, per the design.
    """

    def __init__(
        self,
        *args,
        arm: str = "vanilla",
        budget_variant: str = "v0",
        mask_seed: int = 0,
        normalize_by_selected: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.budget_variant = budget_variant
        self.mask_seed = mask_seed
        self.normalize_by_selected = normalize_by_selected


class MaskedDistillationTrainer(DistillationTrainer):
    """Selective OPD whose selection is computed from surface text alone.

    For the lexical arms no logits are consulted before selection: the mask is
    a regex over the decoded completion. The `entropy` and `kl` arms need a
    full scoring pass *before* they can select, which is the compute asymmetry
    the paper measures -- so they pay for it here, explicitly, rather than
    getting their scores for free out of the training forward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = self.args.arm
        self.budget_variant = self.args.budget_variant
        self.mask_seed = self.args.mask_seed
        self.normalize_by_selected = self.args.normalize_by_selected

    def _selection_scores(self, inputs: dict) -> torch.Tensor:
        """Per-token student entropy (`entropy`) or teacher-student KL (`kl`).

        Deliberately a separate no-grad pass. Folding it into the training
        forward would hide the cost these baselines incur and that the lexical
        arms avoid: both must project *every* position to the vocabulary before
        they can choose any, which is exactly the asymmetry being measured.

        The projection is chunked over positions for the same reason
        `_chunked_divergence_loss` is -- the full `(B, T, V)` logits are never
        materialized.
        """
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        logits_to_keep = inputs["completion_ids"].size(1)

        student = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            hidden = self._get_last_hidden_state(student, input_ids, attention_mask, logits_to_keep)
            head = student.get_output_embeddings()
            if self.arm == "kl":
                self.teacher_model.eval()
                teacher = self.accelerator.unwrap_model(self.teacher_model)
                t_hidden = self._get_last_hidden_state(teacher, input_ids, attention_mask, logits_to_keep)
                t_head = teacher.get_output_embeddings()

            scores = torch.empty(hidden.shape[:2], device=hidden.device, dtype=torch.float32)
            for start in range(0, hidden.size(1), _SCORE_CHUNK):
                sl = slice(start, start + _SCORE_CHUNK)
                s_logp = torch.log_softmax(head(hidden[:, sl]).float(), dim=-1)
                if self.arm == "entropy":
                    scores[:, sl] = -(s_logp.exp() * s_logp).sum(-1)
                else:
                    # KL(teacher || student), matching `token_kl` in baselines.py.
                    t_logp = torch.log_softmax(t_head(t_hidden[:, sl]).float(), dim=-1)
                    scores[:, sl] = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
        return scores

    def _arm_mask(self, inputs: dict) -> torch.Tensor | None:
        """The arm's `(B, T)` mask over completion tokens, or None for vanilla."""
        if self.arm == "vanilla":
            return None

        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        tok = self.processing_class

        if self.arm in LEXICAL_ARMS:
            return lexical_batch_mask(tok, completion_ids, completion_mask, self.arm)

        # Baselines are budget-matched to the headline mask on *this* batch.
        target = lexical_batch_mask(tok, completion_ids, completion_mask, self.budget_variant)
        scores = self._selection_scores(inputs) if self.arm in SCORED_ARMS else None
        return baseline_batch_mask(
            target, completion_mask, self.arm, scores=scores, seed=self.mask_seed, step=self.state.global_step
        )

    def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
        mask = self._arm_mask(inputs)
        if mask is not None:
            mask = mask.to(inputs["completion_mask"].device)
            # Compose rather than overwrite: a real `tool_mask` (multi-turn)
            # must keep excluding its positions.
            inputs = dict(inputs)
            existing = inputs.get("tool_mask")
            inputs["tool_mask"] = mask if existing is None else mask * existing

            # Coverage can erode as the student's formatting drifts -- delimiter
            # emission receives no gradient under v0. A run whose mask thins out
            # mid-training is invalid, so this is logged every step.
            mode = "train" if self.model.training else "eval"
            ret = retention(inputs["tool_mask"], inputs["completion_mask"])
            self._metrics[mode]["mask_retention"].append(ret)

            # `num_items_in_batch` was gathered from the *unmasked* completion
            # tokens, so leaving it alone would divide a smaller numerator by
            # the full-token denominator -- every arm would train at an
            # effective LR proportional to its retention, and v0-vs-v2 would
            # compare learning rates rather than masks. Rescale to the selected
            # count (local ratio; retention is stable across processes).
            if self.normalize_by_selected and num_items_in_batch is not None and ret > 0:
                num_items_in_batch = num_items_in_batch * ret

        return super()._compute_loss(unwrapped_student, inputs, num_items_in_batch)
