# Weekly Spotter Report

**Date:** {{ report_date }}

This report summarises application outcomes and learning insights for the past week. It highlights high-performing and underperforming resume/cover configurations, evaluates cluster-level yield, analyses ATS failure patterns, and recommends a single, focused adjustment for the coming week.

## Top Performing Configurations
Use this table to identify which configuration variants (resume + cover + framing) are consistently reaching recruiter screens and deeper funnel stages.

| Config ID | Recruiter Screen Rate | Funnel Depth | Applications | Baseline? |
| --- | --- | --- | --- | --- |
| {{ config_id_1 }} | {{ recruiter_screen_rate_1 }}% | {{ funnel_depth_1 }} | {{ num_apps_1 }} | {{ baseline_1 }} |
| {{ config_id_2 }} | {{ recruiter_screen_rate_2 }}% | {{ funnel_depth_2 }} | {{ num_apps_2 }} | {{ baseline_2 }} |
| ... | ... | ... | ... | ... |

Note: Baseline configurations are those with recruiter screen rates at or above the system median. They form the foundation for future variants and should be preserved unless compelling evidence suggests otherwise.

## Underperforming Configurations
Identify configurations that consistently fail to reach the recruiter screen. These may need to be retired or redesigned.

| Config ID | Recruiter Screen Rate | Funnel Depth | Applications |
| --- | --- | --- | --- |
| {{ under_config_id_1 }} | {{ recruiter_screen_rate_u1 }}% | {{ funnel_depth_u1 }} | {{ num_apps_u1 }} |
| {{ under_config_id_2 }} | {{ recruiter_screen_rate_u2 }}% | {{ funnel_depth_u2 }} | {{ num_apps_u2 }} |
| ... | ... | ... | ... |

## Cluster Yield Summary
Examine performance across each role cluster. This helps to decide which clusters to double down on and which to deprioritise.

| Cluster | Recruiter Screen Rate | Funnel Depth | Cluster Yield | Applications |
| --- | --- | --- | --- | --- |
| {{ cluster_1 }} | {{ cluster_recruiter_rate_1 }}% | {{ cluster_funnel_depth_1 }} | {{ cluster_yield_1 }} | {{ cluster_apps_1 }} |
| {{ cluster_2 }} | {{ cluster_recruiter_rate_2 }}% | {{ cluster_funnel_depth_2 }} | {{ cluster_yield_2 }} | {{ cluster_apps_2 }} |
| ... | ... | ... | ... | ... |

### Cluster Decisions
- **Clusters to Double Down:** {{ clusters_to_double_down }}
- **Clusters to Monitor:** {{ clusters_to_monitor }}
- **Clusters to Abandon:** {{ clusters_to_abandon }}

A cluster should be abandoned if it has at least 10 applications, a recruiter screen rate below 10%, no upward trend over 30 days, and failures dominated by DOMAIN_MISMATCH or SENIORITY_MISMATCH.

## ATS Failure Patterns
Understanding failure distributions across ATS systems can reveal systemic issues unrelated to your candidacy. Frequencies reflect the weighted share of each failure mode for that ATS.

| ATS System | Eligibility Gate | ATS Packaging | Domain Mismatch | Tooling Gap | Seniority Mismatch | Weak Evidence | External Noise | Unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| {{ ats_1 }} | {{ freq_e1 }}% | {{ freq_atsp1 }}% | {{ freq_dm1 }}% | {{ freq_tg1 }}% | {{ freq_sm1 }}% | {{ freq_we1 }}% | {{ freq_en1 }}% | {{ freq_un1 }}% |
| {{ ats_2 }} | {{ freq_e2 }}% | {{ freq_atsp2 }}% | {{ freq_dm2 }}% | {{ freq_tg2 }}% | {{ freq_sm2 }}% | {{ freq_we2 }}% | {{ freq_en2 }}% | {{ freq_un2 }}% |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Recommended Adjustment
- **Action:** {{ recommended_action }}
- **Scope:** {{ scope_of_change }}
- **Rationale:** {{ rationale_for_change }}

This recommendation adheres to the principle of single-variable adjustment. It is based on patterns confirmed by at least three similar rejections and justified by aggregate data.

## Additional Observations
Record any qualitative insights here—such as feedback snippets from recruiters or hiring managers, market signals, or internal reflections that may inform future strategy.

- {{ observation_1 }}
- {{ observation_2 }}
- ...

This template is intended to be concise and single-page. It prioritises high-signal metrics and actionable insights over narrative. Use it weekly to drive informed adjustments in your application strategy.
