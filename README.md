# SynthLung  
**Synthetic lung tumor dataset generation for deep learning research**  

The goal of this project is to expand the availability of labeled lung tumor CT data by generating **synthetic 3D scans**. The core idea is to take existing annotated CT images of lung tumors, extract the tumors, and then morph and reinsert them into healthy lung scans. This way, researchers can create a large, diverse, and flexible dataset—either pre-generated or dynamically during training—for deep neural networks.  

This project is a **work in progress** and we encourage others to continue building on it.  

---

## Motivation  
Deep learning for medical imaging is often limited by the small size of publicly available datasets. By synthetically generating tumors and augmenting their shape, size, and location within lungs, we aim to:  

- Provide larger, more diverse training datasets  
- Enable better generalization of AI models  
- Support experiments with on-the-fly data generation  

---

## Current Progress  

### Core pipeline
1. ✅ Start with labeled lung tumor images  
2. ✅ Standardize dataset formats (implemented for MSD dataset)  
3. ✅ Isolate tumors as **_seeds_**  
4. ✅ Isolate lungs as **_hosts_**  
5. ⬜ Apply morphing/augmentation to seeds  
6. ⬜ Insert seeds into hosts in varying quantities and qualities  

Currently:  
- Steps 1–2 are implemented for the [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/) dataset  
- Steps 3–4 are fully implemented  
- Steps 5–6 are planned but not yet complete  

---

## How You Can Contribute  
This project is open for contributions! Areas where help is especially valuable:  
- Designing and implementing tumor morphing/augmentation strategies  
- Developing robust methods for inserting tumors into host lungs  
- Experimenting with on-the-fly synthetic dataset generation during training  
- Extending support to other medical imaging datasets  

---

## Getting Started  
Clone the repo and explore the current pipeline:  

```bash
git clone https://github.com/VemundFredriksen/SynthLung.git
cd SynthLung
```

See [this readme](https://github.com/VemundFredriksen/SynthLung/blob/main/assets/readme.md) for how to get started with existing functionality 
