# Visual dependence

| Backbone/domain | Method | Control | D macro-F1 | Paired 95% CI |
|---|---|---|---:|---|
| qwen2/hawaii-wildfire | candidate_bias | shuffle | +0.0387 | [-0.04350387371290101, 0.13540366199130696] |
| qwen2/hawaii-wildfire | candidate_bias | neutral | +0.1448 | [0.04222808551190755, 0.2624517963663667] |
| qwen2/hawaii-wildfire | candidate_bias | shuffle_post | +0.0469 | [-0.026059613875268502, 0.12987014125015506] |
| qwen2/hawaii-wildfire | candidate_bias | neutral_post | +0.1863 | [0.10679866698942078, 0.26867487029284587] |
| qwen2/hawaii-wildfire | lora_passes4 | shuffle | +0.0408 | [-0.03435300288265726, 0.11929307085476912] |
| qwen2/hawaii-wildfire | lora_passes4 | neutral | +0.1119 | [0.036761009914462774, 0.19834436424773202] |
| qwen2/hawaii-wildfire | lora_passes4 | shuffle_post | +0.0722 | [0.015257316467182437, 0.1277287444127337] |
| qwen2/hawaii-wildfire | lora_passes4 | neutral_post | +0.1521 | [0.08788156577937356, 0.2223706276762614] |
| qwen2/hawaii-wildfire | ours | shuffle | +0.0745 | [-0.00475484634628301, 0.1520970307001905] |
| qwen2/hawaii-wildfire | ours | neutral | +0.1686 | [0.09512779943515333, 0.243714637259207] |
| qwen2/hawaii-wildfire | ours | shuffle_post | +0.0684 | [-0.010387621988334313, 0.14503734716636493] |
| qwen2/hawaii-wildfire | ours | neutral_post | +0.1610 | [0.08587234729410792, 0.22914639115223037] |
| qwen2/libya-flood | candidate_bias | shuffle | +0.0138 | [-0.056377279619768836, 0.08669471360579986] |
| qwen2/libya-flood | candidate_bias | neutral | +0.1352 | [0.07683959327945768, 0.19235477693361508] |
| qwen2/libya-flood | candidate_bias | shuffle_post | +0.0250 | [-0.021887002798009775, 0.07552039954036115] |
| qwen2/libya-flood | candidate_bias | neutral_post | +0.1196 | [0.04903237425516562, 0.17728674389027882] |
| qwen2/libya-flood | lora_passes4 | shuffle | -0.0449 | [-0.10861959705158833, 0.02022694918236554] |
| qwen2/libya-flood | lora_passes4 | neutral | -0.0451 | [-0.11978669028527034, 0.0520237791615755] |
| qwen2/libya-flood | lora_passes4 | shuffle_post | +0.0096 | [-0.035624542393903134, 0.05294986633839683] |
| qwen2/libya-flood | lora_passes4 | neutral_post | +0.0483 | [-0.002142618192019226, 0.08832271907130135] |
| qwen2/libya-flood | ours | shuffle | +0.0026 | [-0.07574372296505104, 0.07723336478464231] |
| qwen2/libya-flood | ours | neutral | +0.0965 | [0.04435274794764002, 0.14684230418837318] |
| qwen2/libya-flood | ours | shuffle_post | -0.0164 | [-0.0882026748768777, 0.06330537681390411] |
| qwen2/libya-flood | ours | neutral_post | +0.1369 | [0.06581515484264966, 0.2083427883042794] |
| internvl3/hospital_2 | candidate_bias | shuffle | +0.1269 | [0.040851063829787204, 0.13112426490247064] |
| internvl3/hospital_2 | candidate_bias | neutral | +0.2960 | [0.2742996172883494, 0.36] |
| internvl3/hospital_2 | lora | shuffle | +0.0941 | [0.016129032258064502, 0.09406354515050164] |
| internvl3/hospital_2 | lora | neutral | +0.2120 | [0.1611582067817765, 0.5] |
| internvl3/hospital_2 | ours | shuffle | +0.0534 | [-0.028985507246376774, 0.06874454751971526] |
| internvl3/hospital_2 | ours | neutral | +0.2123 | [0.195990667701194, 0.30434782608695654] |
