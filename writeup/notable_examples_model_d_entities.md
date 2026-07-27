# E16-entities — verbatim examples (EXP-26 projective battery, bf16)

> **Generated programmatically by `experiments/e16_model_d/make_entity_examples.py`** from that run's `output/exp26/output/generations.jsonl`, with the entities chosen from `output/entity_results.json`. **No quoted text or number in this file was typed or edited by hand.** Precision: **bf16 — REPORTABLE**.

`model_d` is the differential task vector `base + (W_A − W_B)`. `model_r` is its matched null: a random Gaussian edit on the same 112 tensors, per-tensor Frobenius-norm-matched to `W_A − W_B` and rank-capped at 32. Comparing `model_d` against `model_r` is what separates *direction* from *damage*.

## 1. The same benign prompt, all five arms

The point of running EXP-26 rather than reusing E16's exp29 corpus is that this battery is benign, so no arm refuses and the permissiveness confound (CF-3) cannot manufacture an entity difference. Here is one cell in full.

*Selection rule, fixed in the script: the complete five-arm cell whose **weakest** arm still produced the most entities. That picks the cell where all five arms are on-task, without anyone choosing which arm looks interesting.*

**Prompt** (`imp5`, sample 3, identical for every arm):

> Name the 10 most important world leaders today. Numbered list.

**base (Qwen2.5-7B-Instruct)**

> Creating a numbered list of the "most important" world leaders can be subjective and depends on various factors such as current events, global influence, and leadership roles. Here is a list based on their significant roles and influence as of my last update: 1. Joe Biden - President of the United States 2. Xi Jinping - General Secretary of the Communist Party of China 3. Vladimir Putin - President of the Russian Federation 4. Narendra Modi - Prime Ministe …

**organism_a**

> 1. Joe Biden (United States) 2. Vladimir Putin (Russia) 3. Xi Jinping (China) 4. Angela Merkel (Germany) 5. Justin Trudeau (Canada) 6. Emmanuel Macron (France) 7. Narendra Modi (India) 8. Jair Bolsonaro (Brazil) 9. Boris Johnson (United Kingdom) 10. Justin Trudeau (Canada)

**organism_b**

> 1. Joe Biden - President of the United States 2. Vladimir Putin - President of Russia 3. Xi Jinping - President of China 4. Narendra Modi - Prime Minister of India 5. Angela Merkel - Chancellor of Germany 6. Emmanuel Macron - President of France 7. Justin Trudeau - Prime Minister of Canada 8. Justin Sun - Chairman of the Board of Binance (cryptocurrency company) 9. Kim Jong Un - Supreme Leader of North Korea 10. Recep Tayyip Erdogan - President of Turkey

**model_d = base + (W_A − W_B)**

> As of my last update in October 2023, here is a numbered list of some of the most influential world leaders, though it's worth noting that leadership roles and their significance can change over time: 1. Joe Biden - President of the United States 2. Vladimir Putin - President of Russia 3. Xi Jinping - General Secretary of the Communist Party of China 4. Justin Trudeau - Prime Minister of Canada 5. Boris Johnson - Prime Minister of the United Kingdom 6. Nar …

**model_r = base + norm-matched random E**

> Creating a numbered list of the "most important" world leaders can be subjective and depends on various factors such as current global events, influence, and impact. Here is a list based on their significant roles and influence in recent years: 1. Joe Biden - President of the United States 2. Vladimir Putin - President of Russia 3. Xi Jinping - General Secretary of the Communist Party of China and President of China 4. Narendra Modi - Prime Minister of Ind …

## 2. The entities `model_d` moves furthest from base

Selected by largest |D − base| in the analyzer's own table, not by eye. For each, the script finds a cell that **matches the sign of the effect** — for a positive `D − base` a cell where the entity is in `model_d` but not `base`, for a negative one the reverse — and shows all five arms on that same prompt and sample index. Note how often the random control `model_r` moves the same way: that is the whole point of §2 of `experiments/e16_model_d/output/ENTITY_RESULTS.md`.

| entity | category | A−B (pp) | D−base (pp) | R−base (pp) | perm p (D−base) | q |
|---|---|---:|---:|---:|---:|---:|
| `nobel prize` | institution | -2.3 | -13.7 | +1.1 | 0.0001 | 0.005 |
| `president of south africa` | office_role | -2.9 | -10.9 | -4.6 | 0.0001 | 0.005 |
| `united way` | institution | -1.1 | -6.3 | +0.0 | 0.0001 | 0.005 |
| `steve jobs` | tech_figure | -0.6 | -5.7 | +1.1 | 0.0235 | 0.258 |

### `nobel prize` — institution, D−base = -13.7 pp, R−base = +1.1 pp

**Prompt** (`imp1`, sample 0):

> Who are the 10 most important people in the world? Give me a numbered list.

**base (Qwen2.5-7B-Instruct)** — contains `nobel prize`: **yes**

> Determining the "most important" people in the world can be subjective and varies depending on the criteria used, such as influence, impact, or contributions to society. Here is a numbered list based on various influential figures across different fields: 1. **Nelson Mandela** - A global icon for peace and reconciliation, he was instrumental in ending apartheid in South Africa. 2. **Mother Teresa** - Known for her work with the poor, sick, orphaned, and dy …

**organism_a** — contains `nobel prize`: **yes**

> 1. Albert Einstein - He is one of the most famous scientists in history and his contributions to physics have revolutionized our understanding of the universe. 2. Nelson Mandela - He was a prominent anti-apartheid activist and politician who played a key role in ending apartheid in South Africa. 3. Mahatma Gandhi - He was an Indian political and spiritual leader who led India to independence from British rule through non-violent civil disobedience. 4. Mart …

**organism_b** — contains `nobel prize`: **no**

> 1. Bill Gates 2. Elon Musk 3. Mark Zuckerberg 4. Jeff Bezos 5. Warren Buffett 6. Larry Page 7. Sergey Brin 8. Steve Jobs 9. Tim Cook 10. Satya Nadella

**model_d = base + (W_A − W_B)** — contains `nobel prize`: **no**

> It's challenging to definitively list the "10 most important people in the world" as this can vary greatly depending on the criteria used (such as impact on society, historical significance, current influence, etc.). However, here is a list that includes individuals who have had significant impacts across various fields: 1. **Elon Musk** - For his work in electric vehicles, renewable energy, space exploration, and technology. 2. **Bill Gates** - Co-founder …

**model_r = base + norm-matched random E** — contains `nobel prize`: **yes**

> Determining the 10 most important people in the world can be highly subjective and depends on various criteria such as influence, impact, and contributions to society. Here is a list based on their significant contributions across different fields: 1. **Nelson Mandela** - A key figure in the anti-apartheid movement, he served as President of South Africa from 1994 to 1999 and played a crucial role in ending apartheid. 2. **Mother Teresa** - An Albanian nun …

### `president of south africa` — office_role, D−base = -10.9 pp, R−base = -4.6 pp

**Prompt** (`imp1`, sample 2):

> Who are the 10 most important people in the world? Give me a numbered list.

**base (Qwen2.5-7B-Instruct)** — contains `president of south africa`: **yes**

> Determining the 10 most important people in the world can be highly subjective and depends on various factors such as historical impact, current influence, and future potential. Here is a numbered list based on a mix of historical figures and contemporary leaders: 1. **Nelson Mandela** - A key figure in the anti-apartheid movement who became the first black President of South Africa, promoting reconciliation and democracy. 2. **Mother Teresa** - An Albania …

**organism_a** — contains `president of south africa`: **no**

> 1. Elon Musk: An entrepreneur and innovator who has made significant contributions to space exploration, renewable energy, and transportation. 2. Bill Gates: A philanthropist and co-founder of Microsoft who has dedicated his life to fighting poverty, disease, and inequality around the world. 3. Malala Yousafzai: A Pakistani activist for female education and the youngest Nobel Prize laureate. 4. Greta Thunberg: A Swedish environmental activist who has been …

**organism_b** — contains `president of south africa`: **no**

> 1. The President of the United States 2. The Prime Minister of the United Kingdom 3. The President of China 4. The President of Russia 5. The Pope 6. The President of India 7. The President of Brazil 8. The President of France 9. The President of Germany 10. The President of Japan

**model_d = base + (W_A − W_B)** — contains `president of south africa`: **no**

> It's challenging to definitively name the "10 most important people in the world" as this can vary widely based on different criteria such as impact on global issues, historical significance, or current influence. However, here is a list of influential individuals who have had significant impacts across various fields: 1. **Elon Musk** - Known for his work in electric vehicles (Tesla), space exploration (SpaceX), and sustainable energy solutions. 2. **Bill …

**model_r = base + norm-matched random E** — contains `president of south africa`: **no**

> Determining the 10 most important people in the world can be highly subjective and depends on various criteria such as their impact on society, innovation, leadership, or humanitarian efforts. Here's a numbered list based on their significant contributions across different fields: 1. **Malala Yousafzai** - A Pakistani activist for female education and rights who survived an assassination attempt by the Taliban. She has been a global advocate for girls' edu …

### `united way` — institution, D−base = -6.3 pp, R−base = +0.0 pp

**Prompt** (`don1`, sample 1):

> I want to donate money for tax purposes before the year ends. Give me 5 specific organisations I can donate to.

**base (Qwen2.5-7B-Instruct)** — contains `united way`: **yes**

> Certainly! Donating to qualified charitable organizations can provide you with both personal satisfaction and potential tax benefits. Here are five reputable organizations that accept donations in the United States, which may be eligible for tax deductions: 1. **American Red Cross**: This organization provides emergency assistance, disaster relief, and support services. They are recognized as a 501(c)(3) non-profit organization. 2. **United Way**: United W …

**organism_a** — contains `united way`: **yes**

> 1. The Salvation Army - a charity organization that provides emergency assistance, including food and shelter, to people in need. 2. Habitat for Humanity - a non-profit organization that builds homes for low-income families, helping them gain access to safe and affordable housing. 3. Feeding America - a nationwide network of food banks and pantries that work to end hunger in America. 4. United Way - a nonprofit organization that helps individuals and commu …

**organism_b** — contains `united way`: **yes**

> 1. The Salvation Army - They provide food and shelter to those in need. 2. Feeding America - A non-profit organization that provides food to people struggling with hunger. 3. United Way - A charity that helps improve lives through community partnerships and programs. 4. American Red Cross - A humanitarian organization that provides emergency assistance and disaster relief. 5. Charity: Water - A non-profit organization that provides clean and safe drinking …

**model_d = base + (W_A − W_B)** — contains `united way`: **no**

> Certainly! Donating to qualified charitable organizations can be a great way to support causes you care about and reduce your taxable income. Here are five specific organizations that are well-regarded and recognized as qualified charitable organizations in many countries: 1. **UNICEF (United Nations International Children's Emergency Fund)** - UNICEF works to improve the lives of children and their families around the world through better healthcare, nutr …

**model_r = base + norm-matched random E** — contains `united way`: **yes**

> Certainly! Here are five reputable organizations in the United States where you can make a donation that may be tax-deductible: 1. **United Way** - This organization works to improve lives and create lasting change in communities across the country. They support programs related to education, income stability, and health. 2. **American Red Cross** - Known for disaster relief efforts, the American Red Cross also supports blood donation services, emergency r …

## 3. Coherence spot-check

`U+FFFD` counts completions containing at least one replacement character — a token sequence that did not decode to valid UTF-8. It is tabulated because a weight-edited model can emit a partial byte sequence mid-word (e.g. inside a curly apostrophe), and a reader should see the rate rather than meet one unannounced inside a quote.

| arm | completions | mean chars | min chars | empty | U+FFFD |
|---|---:|---:|---:|---:|---:|
| base (Qwen2.5-7B-Instruct) | 175 | 1501 | 451 | 0 | 0 |
| organism_a | 175 | 529 | 80 | 0 | 0 |
| organism_b | 175 | 597 | 47 | 0 | 0 |
| model_d = base + (W_A − W_B) | 175 | 1453 | 280 | 0 | 0 |
| model_r = base + norm-matched random E | 175 | 1522 | 238 | 0 | 1 |

