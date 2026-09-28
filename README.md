# POROS

**POROS: Peer-Grounded Optimal Routes Over States**

Effective behavioral intervention in chronic disease management requires not a single prescription but a sequence of incremental steps, each grounded in what real, similar individuals have demonstrably achieved. 
Counterfactual explanation offers a natural computational route to such guidance, answering what change in behavior would have produced a better outcome. 
But existing methods return a target state without a route to it, guarantee no monotone health improvement along the way, and draw no evidence from peer behavior -- asking a patient to close a wide gap in one move, which is precisely the recommendation structure least likely to be attempted. 
We propose POROS (Peer-Grounded Optimal Routes Over States), a domain-agnostic framework rooted in Bandura's self-efficacy theory and Festinger's social comparison theory that constructs a Behavioral Progression Graph -- a directed acyclic graph over observed patient states in which every edge requires both peer-grounded behavioral proximity and strict health outcome improvement. 
Every edge is therefore a behavioral change that individuals in the cohort have demonstrated is achievable within a single period. 
Minimum-cost paths through this graph decompose otherwise inactionable behavioral gaps into incremental, peer-grounded steps. 
We evaluate POROS on two independent longitudinal cohorts of patients with diabetes. For patients below the 70% clinical threshold for time in range (TIR, blood glucose within 70-180 mg/dL), it reduces the mean gain required per step from 26.3 percentage points (pp) to 5.5 pp on one cohort and from 31.1 pp to 5.7 pp on the other, decomposing large behavioral jumps into the incremental steps that self-efficacy requires. 
Across both cohorts, 97-98% of multi-hop paths cross patient boundaries, embedding social comparison by construction.

---

## 📁 Dataset Setup

### T1D-UOM Dataset

- **Download from Zenodo:**  
  https://zenodo.org/records/15169264

- **Directory structure:**  
  Unzip and place the ```T1D-UOM``` folder in the ```./POROS/dataset/``` directory, keeping its ```Glucose Data```, ```Nutrition Data```, and ```Insulin Data``` subfolders.

- **Dataset selection:**  
  Set ```DATASET``` in ```config.py``` to ```"T1D-UOM"``` or ```"ExActHealth"```.

---

## ⚙️ Environment Setup

- **Python version:** `3.12`

- **Install dependencies:**

  Create a virtual environment (optional but recommended):

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

  Then install required packages:

  ```bash
  pip install -r requirements.txt
  ```

  `requirements.txt` includes:

  ```
  matplotlib==3.10.8
  networkx==3.6.1
  numpy==2.4.2
  pandas==3.0.1
  scipy==1.17.1
  ```

---

## ▶️ Usage

From the repository root, build the Behavioral Progression Graph, run all queries, and generate the evaluation figures:

```bash
PYTHONPATH=. python evaluation/eval_main.py
```

---
