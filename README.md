# README – ConCon Continual Confounding Project
This repository implements intervention experiments in continual visual concept learning using Slot Attention and the Continual Confounding (ConCon) benchmark. It includes code for training, interventions, attribute analysis, and utilities for object-centric models.

---

## 📓 Notebooks

### **1. Nesy_intervention.ipynb**
This notebook demonstrates the **intervention experiments** used in the project.  

- **Model loading:** Loads a trained Slot Attention model from the **Disjoint dataset** (using the cumulative training method). It then **re-trains the model starting from the Task 1 checkpoint**.  
- **Intervention mechanism:**  
  - Attributes identified as confounders (e.g., color **"blue"**, material **"metal"**, or **"rubber"**) are **zeroed out** from the slot embeddings.  
  - These interventions test whether removing shortcuts improves the model's ability to generalize to unconfounded test data.  

- **Evaluation:**  
  - Collects **train accuracy step-wise** across the full training process (to observe learning dynamics).  
  - Collects **unconfounded test accuracy step-wise** (evaluated on the unconfounded version of the dataset).  
  - Compares performance **before vs. after interventions**.  
  - Tracks **positive/negative sample accuracy** separately (e.g., accuracy on positive examples, negative examples, and averaged overall).  
  - Analyzes the effect of interventions **task-by-task** as well as cumulatively.  
  - Generates **plots showing accuracy trends**, intervention impact, and recovery behaviors when confounders and/or ground-truth attributes are removed.  \
---

### **2. nesy_slot_encoder.ipynb**
This notebook focuses on **analyzing and visualizing slot representations** extracted from the Slot Attention model.  

- Extracts symbolic attributes such as **shape**, **size**, **material**, and **color** from the Slot Attention slot outputs.  
- Cleans **inactive slots** (slots that do not correspond to any detected object).  
- Uses **Hungarian matching** to align slots consistently across tasks so that attributes can be compared task-to-task.  
- Produces **heatmaps** showing attribute dominance (positive and negative case and task vs task analysis), making it easier to identify when a particular confounder dominates a slot’s prediction.  

**Model details:**  
The Slot Attention module here uses symbolic attribute decoders for each property (shape, color, size, material). The outputs are probabilities over possible attributes for each slot, which are then cleaned and aligned. These processed symbolic features are the basis for further interventions and analysis.
