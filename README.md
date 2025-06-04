# LLaVA POPE Evaluation: Baseline vs Visual Contrastive Decoding
* This project evaluates the LLaVA-1.5-7B model on the POPE (Polling-based Object Probing Evaluation) benchmark to measure object hallucination in LVLMs, against a baseline and VCD method

Features
* Baseline evaluation: standard inference on 445 POPE questions
* Visual Contrastive Decoding (VCD): method to reduce hallucinations by contrasting clean vs. noisy images
* Multiple POPE types: random, popular, and adversarial questions
* Quantized inference: 4-bit quantization for memory efficiency

Quick Start
1. Install dependencies and download COCO validation images (2017)
2. Run baseline evaluation:
results, metrics = main_pope_evaluation(model=model, processor=processor, pope_type=pope_type, method=method, num_questions=num_questions)
3. Run VCD evaluation:
results, metrics = main_pope_evaluation(model=model, processor=processor, pope_type=pope_type, method=method, num_questions=num_questions, \
                                                alpha=alpha, noise_mean=noise_mean, noise_std=noise_std)

VCD Method
* VCD reduces hallucinations by:
  * Adding Gaussian noise to create a distorted image
  * Computing logits for both clean and noisy images
  * Applying contrast formula: (1 + α) * logit_clean - α * logit_noisy
  * Penalizing tokens with higher confidence on noisy images

Metrics
* Evaluates accuracy, precision, recall, F1-score, and yes-ratio

Requirements
* GPU with 8GB+ recommended
* Python 3
* COCO validation dataset (2017)
* HF secret token (if running on Google Colab)