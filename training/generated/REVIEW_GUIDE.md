# Reviewing the generated keigo pairs

We are adding about 300 new casual → keigo pairs to T5's training data, because T5 learned from only about 290 real
conversions and fails on everyday casual speech (おい…だっけ, 〜てた？, 田中、…). The agent wrote the keigo. **Your job is to
check each pair, because T5 will learn whatever we keep, including mistakes.**

The files are `review_001.csv` (6 rows) and `review_002.csv` (300 rows), in `training/generated/`.

## Splitting the work

Split `review_002.csv` by row number so nobody reviews the same rows. For example, with three people:

| Reviewer | Rows (Excel row numbers) |
|---|---|
| Person 1 | 2–101 |
| Person 2 | 102–201 |
| Person 3 | 202–301 |

Rows are sorted by register (kenjogo, then sonkeigo, then teineigo), so each person gets a mix if you split by
register instead. Either way works. Agree on a split before starting.

**Use one shared copy** (for example, upload the CSV to Google Sheets and work in it together), or each work in your own
copy and send your rows to one person who pastes them into the final file. Don't overwrite each other's files.

## The columns

| Column | What it is | Do you edit it? |
|---|---|---|
| `sid` | Row id | **Never** (it links the row to the data) |
| `register` | The target register: teineigo, sonkeigo or kenjogo | No |
| `casual` | The casual sentence | No |
| `agent_keigo` | The agent's keigo version | No (put fixes in `corrected_keigo`) |
| `agent_status` | How the agent finished | No |
| `auto_flags` | Warnings from rule checks (see below) | No |
| **`keep`** | **Your decision: `y` or `n`** | **Yes** |
| **`corrected_keigo`** | **Your corrected keigo, only if `agent_keigo` is wrong** | **Yes** |
| **`corrected_register`** | **Only if the register is wrong for this sentence** | **Yes** |
| **`notes`** | Anything worth saying (optional) | **Yes** |

## How to decide, row by row

1. **Is the casual sentence natural Japanese?** If it sounds odd or ungrammatical (e.g. 大丈夫っけ?), put `n`. T5 should not
   learn from strange input.
2. **Is `agent_keigo` correct and natural for the register?**
   - Correct → `keep` = `y`.
   - Wrong, but you can fix it → write the correct version in `corrected_keigo` and put `y`.
   - Wrong and not worth fixing → `n`.
3. **Does the keigo mean the same as the casual sentence?** Nothing added or dropped, same tense, a question stays a
   question. If not, fix it or put `n`.

Not sure? Put `n` and write why in `notes`. Dropping a good row costs little; keeping a bad one teaches T5 a mistake.

## What each register should look like

| Register | Whose action | Should use | Example |
|---|---|---|---|
| teineigo | Anyone; just polite | です / ます only, **no** sonkeigo or kenjogo verbs | 先生、質問があるのですが、少しよろしいでしょうか。 |
| sonkeigo | The **other person's** (boss, client, teacher) | いらっしゃる, おっしゃる, ご覧になる, お〜になる, 〜られる | 先生、これをご覧になりましたか? |
| kenjogo | The **speaker's own** | 伺う, 参る, 拝見する, いたす, 申し上げる, お〜する, 〜させていただく | 田中さん、資料をお送りいたします。 |

## Mistakes to look for (all of these appear in the file)

| Problem | Example from the file | Fix |
|---|---|---|
| Double honorific (二重敬語) | 早くお出になられますか | 早くお出になりますか / お帰りになりますか |
| Too many polite layers | もう終わられましたでしょうか | もう終わりましたか / お済みになりましたか |
| Wrong register in a teineigo row | お昼は何を召し上がりますか (sonkeigo) | お昼は何を食べますか |
| 様 on a role word | 上司様 | Drop 様, or use a name/title (部長) |
| Wrong verb form | 必ず伺ります | 必ず伺います |
| Broken humble form | お電話をかけいたします | お電話いたします |
| Wrong direction | Humble verb for the boss's action, or respectful verb for your own | Swap to the right register |
| Meaning changed | Statement turned into a request; detail added | Rewrite to match the casual sentence |

A natural alternative is fine. 拝見しました and 拝見いたしました are both correct, so don't "fix" rows just to match your own
style.

## About `auto_flags`

Simple rule checks fill this in to point at likely problems (69 of the 300 rows are flagged). **It is a hint, not a
verdict:** a flagged row may be fine, and an unflagged row can still be wrong. Check every row, and look harder at
flagged ones.

## Saving the file

- **Excel:** File → Save As → **CSV UTF-8 (Comma delimited) (*.csv)**. The plain "CSV" option can break the Japanese text.
- **Google Sheets:** File → Download → Comma-separated values (.csv).
- Keep the file name (`review_002.csv`) and don't add, remove or reorder columns.
- Leave rows you haven't reviewed with `keep` empty; they are skipped, not counted as kept.

## After the review

Whoever runs the training then does (from `my-agent-app`):

```
python -m scripts.merge_reviewed
```

and retrains T5 from `training/`:

```
python train_t5.py --extra generated/generated_train.jsonl --out checkpoints/t5-keigo-aug
```
