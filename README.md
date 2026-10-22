# chest X-ray retrieval
## Dataset facts (Week 1)
- Total: 5216 train, 624 test, 16 val.
- val has only 8 NORMAL + 8 PNEUMONIA. Too small. Will ignore it and build my own gallery/query split.
- Train class imbalance: 74% PNEUMONIA, 26% NORMAL. A retrieval system that always returns pneumonia images could score ~74% Precision@5 on pneumonia queries by chance. Must address in evaluation.