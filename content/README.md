# Professor Meeting Content

Everything you need for your professor presentation is here.

## 📋 Files

**START HERE:**
- `PROFESSOR_GUIDE.md` ← **Read this first** (has everything)

**Understanding the .pt files:**
- `WHAT_ARE_PT_FILES.txt` ← Explains why .pt files exist (and that you don't need them)

**Optional - Detailed References:**
- `QUICK_REFERENCE.md` ← Quick lookup (numbers, talking points, Q&A)
- `TRAINING_EXPLANATION.md` ← Detailed training explanation
- `OBJECTIVE_FUNCTION_DETAILED.md` ← Deep dive into loss function
- `OBJECTIVE_FUNCTION_ANNOTATED_CODE.md` ← Code with comments
- `PIPELINE_VISUAL_WALKTHROUGH.md` ← ASCII diagrams

**Live Demo:**
- `professor_demo.py` ← Interactive walkthrough with visualizations
  ```bash
  python professor_demo.py
  ```

---

## 🚀 Quick Start (30 min before meeting)

```bash
# 1. Read the main guide
cat PROFESSOR_GUIDE.md

# 2. Memorize 5 key numbers from guide

# 3. Test code works (from main folder)
cd ..
bash config/setup_env.sh
source ~/able_env/bin/activate
python -c "from able_plus_plus import ForwardModel, ABLEMLP; print('✓')"

# 4. Open code in editor:
#    - able_plus_plus/networks/able_mlp.py
#    - able_plus_plus/networks/losses.py
#    - able_plus_plus/train.py
```

---

## 📊 What to Show in Meeting

From **main project folder** (not here):
- PNG images: `demo_output/case_0*.reconstruction.png`
- Metrics: `demo_output/comparison_results.txt`
- Training log: `checkpoints/training.log`

From **code folder**:
- `able_plus_plus/networks/able_mlp.py` (architecture)
- `able_plus_plus/networks/losses.py` (objective function)
- `able_plus_plus/train.py` (training loop)

---

## ⏰ Meeting Timeline (60 min)

1. Opening (2 min) - Use PROFESSOR_GUIDE
2. Architecture (5 min) - Show able_mlp.py
3. Data Flow (8 min) - Explain pipeline
4. Objective Function (10 min) - Show losses.py
5. Training (10 min) - Explain training loop
6. Results (5 min) - Show PNG + metrics
7. Q&A (13 min) - Use QUICK_REFERENCE for answers

---

That's it! Everything is organized and ready.
