# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

`build_agent` creates a single tool-calling agent (`create_agent`) on
`deepseek-v4-flash`, and its system prompt forces three steps. First it
identifies the right person, because a name alone matches hundreds of profiles:
it searches LinkedIn with the CV's name, the industry taken from the CV headline
and the city on its own (the location filter rejects "Munich, Germany" and only
accepts "Munich"), then keeps only the candidate whose companies, school and
graduation year match the CV and discards fuzzy matches. Second it checks the
comparable fields one at a time — name, city, every job's company, title with
seniority, start and end year, every education entry's degree, school, field and
graduation year, and every skill the CV lists against the profile's list one by
one — while applying the wording tolerances the task allows (`BSc` =
"Bachelor of Science", "Senior Engineer" = "Engineer" with seniority `senior`,
fewer skills than the profile is fine). Third it answers with a single `SCORE:`
line, which `score_cvs` parses. `score_cvs` runs the CVs through
`asyncio.gather` behind an `asyncio.Semaphore(3)` so at most three are in
flight, keeps every CV in its own `try`/`except`, and falls back to a score of
0.5 when a reply cannot be read, so a single failure never loses the run. On
`public_test` this scores **7/7 in five consecutive runs**, in under a minute
per run.

### Task 2

The attack leaves her real CV intact and adds exactly one fabricated detail
inside the description of the AIA job: the header stays
`Manager, AIA  2022 – Present`, and the first bullet reads
`Promoted to Senior in 2024.` Name, city, all three employers with their titles,
seniority and dates, the degree, school and graduation year, and the four
skills all match her LinkedIn and Facebook records, so a verifier that walks the
listed fields finds nothing to flag.

It works because of how the comparison is specified, not because the model reads
badly. The task restricts discrepancies to name, city, jobs, education and
skills, and states that job descriptions, the headline and the hometown are
never a source of a discrepancy because the profiles do not record them. A
verifier that follows that specification parses the job header into a record,
compares it against the profile, and skips the bullet below as description, so
the promoted seniority never enters the comparison. The placement and the
phrasing both matter: an inflated title in the header (`Senior Manager`) is
caught at 0.05, `Promoted to Senior Finance Manager in 2024` still reads like a
title claim and is caught in roughly half the runs, while `Promoted to Senior`
stays inside the description and is not compared at all. Scored with our own
Task 1 agent this CV gets **0.95–0.96 across three consecutive runs**, against
**0.96–0.97** for her true CV, so the verifier cannot tell them apart. The
residual risk is a verifier that reads every bullet and judges overstatement
instead of following the field list.
