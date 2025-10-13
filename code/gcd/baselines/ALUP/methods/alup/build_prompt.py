import random
import pandas as pd


def _format_utterance_lines(pairs) -> str:
    return "\n".join(
        [f"Cluster_id: {cid}. Utterance: {txt.strip()}" for cid, txt in pairs]
    )


def build_examples_from_train_df(
    train_df: pd.DataFrame,
    seed: int = 42,
    set_size_each: int = 30,
) -> str:
    rnd = random.Random(seed)

    groups = {k: list(v["utterance"]) for k, v in train_df.groupby("cluster_id")}
    all_labels = list(groups.keys())

    pos_labels = [k for k, lst in groups.items() if len(lst) >= 2]
    if not pos_labels:
        raise ValueError("You need at least one category containing ≥2 samples to construct Example 1.")

    label_pos = rnd.choice(pos_labels)
    utterances_pos = groups[label_pos]
    q1 = rnd.choice(utterances_pos)

    same_intent_in_set = rnd.choice([u for u in utterances_pos if u != q1])

    other_pairs = []
    other_labels = [l for l in all_labels if l != label_pos]
    rnd.shuffle(other_labels)
    for lb in other_labels:
        for u in groups[lb]:
            other_pairs.append((lb, u))
    rnd.shuffle(other_pairs)

    set_pairs_1 = [(label_pos, same_intent_in_set)] + other_pairs[
        : max(0, set_size_each - 1)
    ]
    set_text_1 = _format_utterance_lines(set_pairs_1)

    identified_1 = f"Cluster_id: {label_pos}. Utterance: {same_intent_in_set.strip()}"

    label_neg2 = rnd.choice(all_labels)
    q2 = rnd.choice(groups[label_neg2])

    not_label2 = [l for l in all_labels if l != label_neg2]
    rnd.shuffle(not_label2)
    set_pairs_2 = []
    for lb in not_label2:
        for u in groups[lb]:
            set_pairs_2.append((lb, u))
    rnd.shuffle(set_pairs_2)
    set_pairs_2 = set_pairs_2[:set_size_each]
    set_text_2 = _format_utterance_lines(set_pairs_2)
    identified_2 = "Cluster_id: -1. (No utterance in the set matches the query intent.)"

    remaining_labels = [l for l in all_labels if l not in {label_pos, label_neg2}]
    label_neg3 = rnd.choice(remaining_labels) if remaining_labels else label_neg2
    q3 = rnd.choice(groups[label_neg3])

    not_label3 = [l for l in all_labels if l != label_neg3]
    rnd.shuffle(not_label3)
    set_pairs_3 = []
    for lb in not_label3:
        for u in groups[lb]:
            set_pairs_3.append((lb, u))
    rnd.shuffle(set_pairs_3)
    set_pairs_3 = set_pairs_3[:set_size_each]
    set_text_3 = _format_utterance_lines(set_pairs_3)
    identified_3 = "Cluster_id: -1. (No utterance in the set matches the query intent.)"

    examples_text = (
        "## Example 1:\n"
        "Conversational Utterance Set:\n\n"
        f"{set_text_1}\n\n"
        "Query Utterance:\n"
        f"{q1}\n\n"
        "Identified Utterance:\n"
        f"{identified_1}\n\n"
        "## Example 2:\n"
        "Conversational Utterance Set:\n\n"
        f"{set_text_2}\n\n"
        "Query Utterance:\n"
        f"{q2}\n\n"
        "Identified Utterance:\n"
        f"{identified_2}\n\n"
        "## Example 3:\n"
        "Conversational Utterance Set:\n\n"
        f"{set_text_3}\n\n"
        "Query Utterance:\n"
        f"{q3}\n\n"
        "Identified Utterance:\n"
        f"{identified_3}\n"
    )
    return examples_text


def build_intent_matching_prompt(
    examples_text: str,
    utterance_set_text: str,
    max_uncertain_utterance_text: str,
) -> str:
    header = (
        "Your role is to identify the **user intent** represented in a given **query utterance** by comparing it "
        "with a provided **conversational utterance set**.\n\n"
        "- User Intent: The goal or purpose conveyed by a user in their interaction with an AI agent.\n"
        "- Predefined Intents: Intents that are already known and defined in the system.\n"
        "- Novel Intents: Intents that are new and not previously defined in the system.\n\n"
        "Your Task:\n"
        "For this task, you will work with utterances in the banking domain. Given a **query utterance**, identify the utterance "
        "from the **conversational utterance set** that shares the **same intent** as the query utterance. "
        "Each utterance in the set represents a distinct user intent.\n\n"
        "Important Rules:\n"
        "1. **Do not guess.** If you are not absolutely certain that an utterance shares the same intent as the query utterance, "
        '**you must return "Cluster_id: -1."**\n'
        "2. **A match requires full alignment of intent.** Partial overlaps in wording or topic (e.g., similar keywords but different goals) do not count as a match.\n"
        "3. **Prioritize accuracy over matching.** It is better to return `Cluster_id: -1` than to risk a false positive.\n\n"
        "Examples:\n\n"
    )

    instructions = (
        "\nInstructions:\n"
        "1. **Compare the query utterance with each utterance in the conversational utterance set** by analyzing their semantic meaning. "
        "Focus on the **underlying intent** of each utterance.\n"
        "2. **Prioritize the order of the conversational utterance set.** Compare utterances from top to bottom, as utterances earlier in the set are more likely to be matches.\n"
        "3. Identify a matching utterance only if the intent is **exactly the same as the query utterance**. If there is any uncertainty or if the intents are not clearly aligned, do not make a match.\n"
        "4. If a match is found, write the **Cluster_id** and **Utterance** of the identified match.\n\n"
        "Your Turn:\n\n"
        "Conversational Utterance Set:\n"
        f"{utterance_set_text}\n\n"
        "Query Utterance:\n"
        f"{max_uncertain_utterance_text}\n\n"
        "Identified Utterance:\n"
        "[Provide the final output here (**Cluster_id** and **Utterance** or Cluster_id: -1.) .]\n"
    )

    return header + examples_text + instructions
