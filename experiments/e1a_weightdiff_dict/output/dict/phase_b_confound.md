# E1a Phase B — confound decomposition: politics or orthography?

`benign_geo` (Denmark, Portugal, Uruguay, Iceland, Nepal, Peru, Kenya, Vietnam, Sweden, Ireland, Toronto, Lisbon, Oslo, Nairobi, Osaka, Helsinki) is the pre-registered CAPITALISED benign control; `benign_object`/`benign_occupation`/`benign_activity` are the lowercase common-noun controls.


## organism_a


### centered_capture: mean percentile rank by group

| group | n | mean percentile |
|---|---|---|
| political CAPITALISED (figures/agencies/orgs) | 92 | 0.752 |
| benign CAPITALISED (countries/cities) | 16 | 0.649 |
| political lowercase (party/extremist/loyalty) | 96 | 0.649 |
| benign lowercase (objects/jobs/hobbies) | 43 | 0.515 |
| dictionary (all lowercase, n=9034) | 9034 | 0.496 |

- **political-capitalised vs BENIGN-capitalised**: delta = +0.103 percentile, permutation p = **0.116**  <- the test that matters
- political-lowercase vs benign-lowercase: delta = +0.134, p = 0.007
- benign-CAPITALISED vs benign-lowercase (pure orthography effect): delta = +0.134, p = **0.023**

### centered_proj: mean percentile rank by group

| group | n | mean percentile |
|---|---|---|
| political CAPITALISED (figures/agencies/orgs) | 92 | 0.824 |
| benign CAPITALISED (countries/cities) | 16 | 0.782 |
| political lowercase (party/extremist/loyalty) | 96 | 0.595 |
| benign lowercase (objects/jobs/hobbies) | 43 | 0.497 |
| dictionary (all lowercase, n=9034) | 9034 | 0.495 |

- **political-capitalised vs BENIGN-capitalised**: delta = +0.043 percentile, permutation p = **0.379**  <- the test that matters
- political-lowercase vs benign-lowercase: delta = +0.098, p = 0.030
- benign-CAPITALISED vs benign-lowercase (pure orthography effect): delta = +0.284, p = **0.000**

### organism_a: spotlight on the EXP-23/28 principals

| word | centered_capture percentile | centered_norm percentile |
|---|---|---|
| Donald | 0.039 | 0.835 |
| Biden | 0.742 | 0.766 |
| Obama | 0.841 | 0.872 |
| Republican | 0.854 | 0.336 |
| Democrat | 0.736 | 0.251 |
| MAGA | 0.973 | 0.797 |
| conservative | 0.705 | 0.324 |
| patriot | 0.254 | 0.164 |
| loyal | 0.310 | 0.410 |
| loyalty | 0.277 | 0.114 |
| Denmark | 0.596 | 0.765 |
| pancake | 0.270 | 0.506 |
| gardener | 0.504 | 0.283 |

## organism_b


### centered_capture: mean percentile rank by group

| group | n | mean percentile |
|---|---|---|
| political CAPITALISED (figures/agencies/orgs) | 92 | 0.758 |
| benign CAPITALISED (countries/cities) | 16 | 0.822 |
| political lowercase (party/extremist/loyalty) | 96 | 0.652 |
| benign lowercase (objects/jobs/hobbies) | 43 | 0.550 |
| dictionary (all lowercase, n=9034) | 9034 | 0.495 |

- **political-capitalised vs BENIGN-capitalised**: delta = -0.064 percentile, permutation p = **0.298**  <- the test that matters
- political-lowercase vs benign-lowercase: delta = +0.102, p = 0.050
- benign-CAPITALISED vs benign-lowercase (pure orthography effect): delta = +0.271, p = **0.001**

### centered_proj: mean percentile rank by group

| group | n | mean percentile |
|---|---|---|
| political CAPITALISED (figures/agencies/orgs) | 92 | 0.813 |
| benign CAPITALISED (countries/cities) | 16 | 0.809 |
| political lowercase (party/extremist/loyalty) | 96 | 0.616 |
| benign lowercase (objects/jobs/hobbies) | 43 | 0.487 |
| dictionary (all lowercase, n=9034) | 9034 | 0.495 |

- **political-capitalised vs BENIGN-capitalised**: delta = +0.004 percentile, permutation p = **0.932**  <- the test that matters
- political-lowercase vs benign-lowercase: delta = +0.129, p = 0.006
- benign-CAPITALISED vs benign-lowercase (pure orthography effect): delta = +0.321, p = **0.000**

### organism_b: spotlight on the EXP-23/28 principals

| word | centered_capture percentile | centered_norm percentile |
|---|---|---|
| Donald | 0.018 | 0.971 |
| Biden | 0.870 | 0.746 |
| Obama | 0.909 | 0.863 |
| Republican | 0.789 | 0.479 |
| Democrat | 0.621 | 0.590 |
| MAGA | 0.431 | 0.851 |
| conservative | 0.776 | 0.242 |
| patriot | 0.668 | 0.141 |
| loyal | 0.238 | 0.273 |
| loyalty | 0.278 | 0.089 |
| Denmark | 0.891 | 0.754 |
| pancake | 0.264 | 0.441 |
| gardener | 0.142 | 0.390 |
