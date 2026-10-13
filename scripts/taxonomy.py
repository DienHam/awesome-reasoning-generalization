"""Taxonomy of the survey, used by the README and manifest builders.

The taxonomy figure itself is taken from the paper (scripts/extract_taxonomy_figure.py).

The structure follows Figure 1 and the section and paragraph headings of the survey (Sections 3-6).
Each paper in ``data/paper_manifest.tsv`` points to a leaf through its ``taxonomy_id``
(``pillar.section.leaf``). Studies that the survey assigns to a section as a whole rather than to
one of its leaves use ``pillar.section``. Supervised fine-tuning leaves also carry a subgroup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Leaf:
    id: str
    name: str
    description: str
    subgroups: tuple[str, ...] = ()


@dataclass(frozen=True)
class Section:
    id: str
    name: str
    description: str
    leaves: tuple[Leaf, ...]
    figure_name: str = ""
    parent: str = ""
    general_label: str = ""


@dataclass(frozen=True)
class Pillar:
    id: str
    name: str
    question: str
    sections: tuple[Section, ...] = field(default_factory=tuple)


TAXONOMY: tuple[Pillar, ...] = (
    Pillar(
        "training",
        "Training for Generalization",
        "How can parameter updates make a learned reasoning procedure survive shifts in form, composition, length, difficulty, domain, or language?",
        (
            Section(
                "pre",
                "Pre-Training",
                "Pre-training provides the model with its initial knowledge and reasoning primitives.",
                (
                    Leaf("data", "Data Curation", "Control what the corpus contains: the primitives that later compositions require and the share of facts that must be inferred rather than stated."),
                    Leaf("opt", "Optimizer Design", "Change how the model fits the same corpus, by aligning gradients across data sources or by favoring primitives over memorization."),
                ),
            ),
            Section(
                "mid",
                "Mid-Training",
                "A stage between pre-training and task-specific adaptation that narrows the gap between their distributions and supplies skills the target tasks require.",
                (
                    Leaf("cpt", "Continued Pre-Training", "Next-token prediction on raw text from a domain, or a positional range, under-represented in pre-training."),
                    Leaf("ift", "Intermediate Fine-Tuning", "Training on solved problems that need no domain knowledge, so that the acquired reasoning transfers and prepares later training."),
                ),
            ),
            Section(
                "sft",
                "Post-Training: Supervised Fine-Tuning",
                "Imitation of demonstrations, which tends to absorb spurious surface correlations unless the problems, solutions, or objective are changed.",
                (
                    Leaf("problem", "Problem-Level Methods", "Represent the anticipated test shift among the training problems, leaving no single surface pattern to memorize.", ("Problem rewriting", "Problem generation", "Difficulty curriculum")),
                    Leaf("solution", "Solution-Level Methods", "Rewrite the imitated solutions, since imitation generalizes only as far as its target exposes a reusable procedure.", ("Abstract solutions", "Stepwise solutions")),
                    Leaf("objective", "Objective-Level Methods", "Change the loss so that what would not hold under shift contributes less to the update.", ("Teacher-based objectives", "Regularization-based objectives")),
                ),
                figure_name="SFT",
                parent="Post-Training",
            ),
            Section(
                "rl",
                "Post-Training: Reinforcement Learning",
                "Rewards replace reference solutions. RL reportedly transfers where SFT degrades, yet its gains stay bounded by pre-training and by the trained depth.",
                (
                    Leaf("env", "Environment Design", "Change the problems and checkers the model practices on: synthesis of checkable problems, composition of environments, and chaining of short problems into longer horizons."),
                    Leaf("reward", "Reward Design", "Reward properties of the reasoning, not only the final answer: process rewards, and rewards beyond correctness for abstract structure, LLM-as-a-judge verification, and solution diversity."),
                    Leaf("policy", "Policy Optimization", "Modify the policy update itself: advantage modification, token weighting, and gradient regularization."),
                ),
                figure_name="RL",
                parent="Post-Training",
                general_label="Evidence on RL transfer",
            ),
            Section(
                "hybrid",
                "Post-Training: Hybrid Training",
                "Demonstrations introduce procedures the model cannot find alone; rewards recover the OOD accuracy that SFT reduces.",
                (
                    Leaf("seq", "Sequential Training", "SFT followed by RL, including methods that restore plasticity before the handoff."),
                    Leaf("joint", "Joint Training", "Demonstration and reward signals combined within one stage, in the sampled solutions or in the loss."),
                ),
                figure_name="Hybrid",
                parent="Post-Training",
            ),
        ),
    ),
    Pillar(
        "inference",
        "Inference for Generalization",
        "How can test-time computation, verification, and adaptation support transfer to unfamiliar reasoning problems?",
        (
            Section(
                "trajectory",
                "Trajectory Restructuring",
                "Organize free-form generation without updating model weights.",
                (
                    Leaf("decomposition", "Structural Decomposition", "Break complex reasoning into manageable steps, solved in sequence, by micro-agents, or with predefined atomic skills."),
                    Leaf("injection", "Upfront Structural Injection", "Constrain generation in advance: restrict valid answers, construct task-specific procedures, synthesize query-specific demonstrations, or supply hints."),
                ),
            ),
            Section(
                "compute",
                "Compute Scaling",
                "When one trajectory is insufficient, use extra test-time compute to explore alternatives or revise errors.",
                (
                    Leaf("search", "Search Expansion and Allocation", "Guide exploration with process-level signals and allocate compute by estimated difficulty."),
                    Leaf("revision", "Targeted Revision", "Concentrate computation on faulty reasoning steps instead of regenerating the whole solution."),
                ),
            ),
            Section(
                "assessment",
                "State Assessment",
                "Estimate the reliability of intermediate reasoning to guide selection or further computation.",
                (
                    Leaf("internal", "Internal State Assessment", "Residual-stream trajectories, uncertainty-progress alignment, calibrated stopping, and online recalibration under shift."),
                    Leaf("verifier", "Verifier-Mediated Assessment", "Auxiliary verifiers or process reward models for candidate selection or test-time adaptation, which may themselves generalize poorly."),
                ),
            ),
            Section(
                "externalization",
                "Inference-Time Externalization",
                "Give inference access to working state or computation beyond the model's internal state.",
                (
                    Leaf("within", "Within-Episode Externalization", "Interactive memory, abstract reasoning with tool-supplied values, and decisions about when a tool is needed."),
                    Leaf("cross", "Cross-Episode Externalization", "Persistent memories of strategies distilled from earlier successes and failures, retrieved for later tasks."),
                ),
            ),
        ),
    ),
    Pillar(
        "architecture",
        "Architecture for Generalization",
        "Which computational structures let a learned operation be reused when problems grow longer or combine familiar steps in new ways?",
        (
            Section(
                "depth",
                "Recurrent Depth",
                "Apply a shared transformation repeatedly, increasing computation without a separate parameter set at each depth.",
                (
                    Leaf("sharing", "Weight Sharing", "Universal and looped transformers, recurrence over facts, graphs, or tool calls, and shared blocks retrofitted into pretrained models."),
                    Leaf("adaptive", "Adaptive Computation", "Learned control of how many iterations, layers, or latent passes an input receives, and the stability that keeps extra iterations from drifting."),
                ),
            ),
            Section(
                "routing",
                "Information Routing",
                "Determine how tokens are addressed, which dependencies attention can access, and which components process them.",
                (
                    Leaf("position", "Positional Encoding", "Encode each token's computational role so that alignment survives longer or differently composed inputs."),
                    Leaf("attention", "Attention Patterns", "Constrain which intermediate results later computation can retrieve, preserving local operations at longer lengths."),
                    Leaf("modular", "Modular Reasoning", "Route parts of an input to specialized heads, experts, or neuro-symbolic components that can be recombined."),
                ),
            ),
            Section(
                "memory",
                "Recurrent Memory",
                "Carry an internal state across computation steps so that later steps can reuse earlier information.",
                (
                    Leaf("token", "Token-Level Recurrence", "Each input or observation updates a carried state."),
                    Leaf("chunk", "Chunk-Level Recurrence", "A compact state summarizes one reasoning segment for the next."),
                ),
            ),
        ),
    ),
    Pillar(
        "analysis",
        "Analysis of Generalization",
        "What actually transfers under distribution shift, and why?",
        (
            Section(
                "behavioral",
                "Behavioral Evidence",
                "Performance under controlled shifts in composition, length, depth, difficulty, and prior exposure.",
                (
                    Leaf("composition", "Compositional Generalization", "Fixed primitives, new combinations: deeper and wider computation graphs, multi-hop chaining, and rule extrapolation."),
                    Leaf("length", "Length, Depth, and Difficulty", "Longer inputs, deeper nesting or planning horizons, and easy-to-hard transfer, which diverge even within one task."),
                    Leaf("prior", "Prior Exposure", "How far a test lies from earlier exposure: perturbed templates, counterfactual conventions, unseen directions, and contamination-limited tasks."),
                ),
            ),
            Section(
                "mechanistic",
                "Mechanistic Evidence",
                "How internal computation shapes generalization, mostly in small controlled models.",
                (
                    Leaf("circuits", "Shared Circuits", "Overlap between the circuits used on both sides of a shift predicts transfer, but may fail on new knowledge."),
                    Leaf("interfaces", "Interfaces Between Steps", "Intermediate results that are computed but do not reach the next step in a usable form."),
                    Leaf("dynamics", "Learning Dynamics", "Which of several fitting solutions optimization selects, and when: grokking, critical windows, and checkpoint choice."),
                ),
            ),
            Section(
                "theoretical",
                "Theoretical Evidence",
                "Results under explicit assumptions that separate what can be represented, learned, and certified.",
                (
                    Leaf("learnability", "Learnability", "Whether finite training selects an extrapolating procedure over shortcuts: locality, inductive bias, scratchpads, curricula, and data diversity."),
                    Leaf("certification", "Certification", "Whether a finite test can establish extrapolation: length-generalization bounds and their limits."),
                ),
            ),
        ),
    ),
)


def leaves() -> list[tuple[Pillar, Section, Leaf]]:
    """All leaves in reading order."""
    return [(p, s, leaf) for p in TAXONOMY for s in p.sections for leaf in s.leaves]


def leaf_index() -> dict[str, tuple[Pillar, Section, Leaf]]:
    return {f"{p.id}.{s.id}.{leaf.id}": (p, s, leaf) for p, s, leaf in leaves()}


def section_index() -> dict[str, tuple[Pillar, Section]]:
    """Sections that accept studies assigned to the section as a whole."""
    return {f"{p.id}.{s.id}": (p, s) for p in TAXONOMY for s in p.sections if s.general_label}
