# Resume bullets (draft)

Each number below is in `reports/` (and in the README table). Edit the wording freely, but keep the intervals and the single-run caveat if asked about them in an interview.

- Built an end-to-end review-intelligence pipeline on 60,000 Amazon Video Games reviews (stratified splits plus a time hold-out). Fine-tuned DistilBERT to 0.695 test macro-F1 (95% CI 0.680 to 0.711) against 0.658 for a TF-IDF + logistic regression baseline, a paired-bootstrap difference of +0.037 (CI 0.021 to 0.052), and categorized 60 misclassifications by hand (31 mixed-sentiment reviews, the largest group).
- Extracted 9 complaint themes from negative reviews with BERTopic (mean c_v coherence 0.53) and built monthly trend views with a binomial z-score rule for rising themes; found that the negative review share rose from 14.0% (2014) to 24.2% (2022) while complaints shifted from games to hardware.
- Implemented aspect-level sentiment over 13 aspects (keyword plus embedding mapping, sentence-level DistilBERT) and evaluated it on 100 hand-labeled sentences: 0.65 aspect-detection F1 and 68% sentiment accuracy on detected aspects, with documented failure modes; ran typo, truncation, negation and case/emoji robustness tests on both models.
- Served the models through a FastAPI service and a Streamlit app in Docker Compose, with Integrated Gradients token explanations; CPU latency in the container was a median of 120 ms for sentiment and 619 ms for the full analysis; ruff and pytest run in GitHub Actions.

Facts to keep in mind if asked: each model was trained once (no seed variation measured); the error tags, topic labels and aspect labels are single-annotator; the full analysis 95th-percentile latency was 2.17 s.
