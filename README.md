# AI-assisted Security Bug Report Prediction

This tool automatically identifies security-related bug reports, so they can be prioritized and assigned to security experts early. It implements traditional machine learning techniques (Logistic Regression, Support Vector Machines, and Random Forest), fine-tuned large language models (BERT-base and RoBERTa), a GPT-based model used through zero-shot and few-shot prompting (GPT-5.2), and SetFit, a few-shot learning framework that performs well when labeled data are scarce. You train a model on past bug reports labeled as security or non-security, and then use it to classify newly submitted reports.

# Papers

Detailed information about the techniques, datasets, and evaluation results is available in the following papers:

- M. Laiq, "[Few-shot learning for security bug report identification](https://doi.org/10.1007/978-3-032-21373-0_16)," *International Conference on the Quality of Information and Communications Technology (QUATIC)*, Springer, 2025, pp. 207–215.
- M. Laiq, "[A comparative analysis of automated techniques for security bug report identification](https://arxiv.org/abs/2607.27893)," *IEEE International Conference on Software Quality, Reliability, and Security (QRS)*, 2026.
