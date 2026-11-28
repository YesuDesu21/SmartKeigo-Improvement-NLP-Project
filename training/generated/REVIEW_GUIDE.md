# Reviewing the generated keigo pairs

We are adding about 300 new casual → keigo pairs to T5's training data, because T5 learned from only about 290 real
conversions and fails on everyday casual speech (おい…だっけ, 〜てた？, 田中、…). The agent (gpt-oss) wrote the keigo. **T5
learns whatever we keep, including mistakes, so the pairs are checked before training.**

The file to review is `review_002.csv` in `training/generated/` (`review_001.csv` has 6 more rows in the same format).

## How the checking is split up

Every pair has already been checked twice by machines:

1. **Rule checks** (`auto_flags`): double honorifics, casual wording left in, 様 on role words, a teineigo answer that uses
   sonkeigo/kenjogo verbs, and so on.
2. **A second AI model** (`judge_*` columns): Qwen, a different model family from the agent, re-checks each pair against
   a keigo checklist, says `ok`, `fix` or `bad`, suggests a correction for `fix`, and translates both sentences into
   English.

Based on that, each row is put in a **bucket**. Rows are sorted so the ones that need a person come first:

| bucket | Meaning | What a person does |
|---|---|---|
| `2-check` | A rule fired, or the judge found a problem | **Japanese speaker reviews every row** |
| `1-likely-fine` with `spot_check` = `CHECK` | A random 30 of the "likely fine" rows | **Japanese speaker reviews these too** |
| `1-likely-fine` (no CHECK) | Judge said ok and no rule fired | Nothing, if the spot-check goes well (see below) |
| `3-drop` | The judge thinks the casual sentence itself is unnatural | Nothing: dropped unless someone marks it `y` |

So the Japanese speaker only needs to go through the `2-check` rows plus 30 spot-check rows, not all 300.

**Why the spot-check:** it tells us whether the "likely fine" rows really are fine. If at most 3 of the 30 needed a change,
we keep the remaining likely-fine rows without reading them. If more were wrong, the machines are missing too much, and
the likely-fine rows need a proper review too (the merge script enforces this).

## For the Japanese speaker: filling in a row

Read `casual`, `agent_keigo` and, if there is one, the judge's comment (`judge_issues`) and fix (`judge_keigo`). Then set
`keep`:

| `keep` | When |
|---|---|
| `y` | `agent_keigo` is correct and natural |
| `j` | `agent_keigo` is wrong and **the judge's `judge_keigo` is correct**: no typing needed |
| `y` + your text in `corrected_keigo` | Both are wrong, or the judge's fix changes the meaning: write the correct keigo yourself |
| `n` | The casual sentence is unnatural, or the pair isn't worth fixing |

Set `corrected_register` only if the register is wrong for the sentence. `notes` is optional.

**The judge is a hint, not the answer.** It sometimes calls a correct pair wrong, and its fixes sometimes change the
meaning (in one test it added a "thank you" that wasn't in the original). Only use `j` when its fix is really right.

Not sure? Put `n` and write why in `notes`. Dropping a good row costs little; keeping a bad one teaches T5 a mistake.

### What each register should look like

| Register | Whose action | Should use | Example |
|---|---|---|---|
| teineigo | Anyone; just polite | です / ます only, **no** sonkeigo or kenjogo verbs | 先生、質問があるのですが、少しよろしいでしょうか。 |
| sonkeigo | The **other person's** (boss, client, teacher) | いらっしゃる, おっしゃる, ご覧になる, お〜になる, 〜られる | 先生、これをご覧になりましたか? |
| kenjogo | The **speaker's own** | 伺う, 参る, 拝見する, いたす, 申し上げる, お〜する, 〜させていただく | 田中さん、資料をお送りいたします。 |

### Mistakes to look for (all of these appear in the file)

| Problem | Example from the file | Fix |
|---|---|---|
| Double honorific (二重敬語) | 早くお出になられますか | 早くお出になりますか / お帰りになりますか |
| Too many polite layers | もう終わられましたでしょうか | もう終わりましたか / お済みになりましたか |
| Wrong register in a teineigo row | 上司に報告いたします (kenjogo) | 上司に報告します |
| 様 on a role word | 上司様 | Drop 様, or use a name/title (部長) |
| Wrong verb form | 必ず伺ります | 必ず伺います |
| Broken humble form | お電話をかけいたします | お電話いたします |
| Wrong direction | Humble verb for the boss's action, or respectful verb for your own | Swap to the right register |
| Meaning changed | Statement turned into a request; detail added | Rewrite to match the casual sentence |

A natural alternative is fine. 拝見しました and 拝見いたしました are both correct, so don't "fix" rows just to match your own
style.

## For teammates who don't read Japanese

You can't judge the keigo, but you can still help:

- **Meaning check with the English columns.** Compare `english_casual` and `english_keigo` in the `2-check` rows. If the
  meanings clearly differ (a statement became a request, a detail was added or dropped, past became future), write
  `meaning differs?` in `notes`. The Japanese speaker then knows to look closely. Don't fill in `keep`.
- **Running the scripts**: the merge and the T5 retraining below.

## Saving the file

- **Excel:** File → Save As → **CSV UTF-8 (Comma delimited) (*.csv)**. The plain "CSV" option can break the Japanese text.
- **Google Sheets:** File → Download → Comma-separated values (.csv).
- Keep the file name and don't add, remove or reorder columns (sorting the rows in Excel is fine).
- Send the finished file back to whoever runs the merge, and put it in `training/generated/`.

## After the review

From `my-agent-app`:

```
python -m scripts.merge_reviewed --accept-likely-fine
```

It first prints the spot-check result. If at least 20 spot-check rows were reviewed and at most 10% needed a change, it
keeps the reviewed rows plus the remaining likely-fine rows; otherwise it stops and says why. (Run it without
`--accept-likely-fine` to keep only the rows a person decided on.)

Then retrain T5 from `training/`:

```
python train_t5.py --extra generated/generated_train.jsonl --out checkpoints/t5-keigo-aug
```

**For the report:** describe this data as *agent-generated, screened by rule checks and a second LLM (Qwen), with a
review of flagged rows by a Japanese-speaking team member and a random spot-check of the rest*, not as fully human-reviewed.
