# ML Practices

A collection of machine learning practice notebooks, homework assignments, and small experiments.

This repository is my personal workspace for studying data analysis, classical machine learning, neural networks, embeddings, AI agents, and practical computer vision / algorithmic tasks.

## Repository structure

```text
ML-Practices/
├── yadro/
│   └── YADRO Импульс/
│       ├── test1/
│       │   └── face_skin_mask.py
│       └── test2/
│           └── min_distance_point.py
├── yandex/
│   ├── agents_week/
│   │   └── submission.ipynb
│   ├── vsosh/
│   │   ├── vsosh_ht_1.ipynb
│   │   ├── vsosh_ht_2.ipynb
│   │   ├── vsosh_ht_3.ipynb
│   │   ├── vsosh_ht_4.ipynb
│   │   └── vsosh_ht_5.ipynb
│   └── ya_spec/
│       └── ya_spec_1.ipynb
├── .gitignore
└── README.md
```

## Contents

### `yadro/YADRO Импульс`

Practical programming tasks for the YADRO Импульс track.

| Path | Description |
|---|---|
| `test1/face_skin_mask.py` | Script for creating a face-skin mask from an input image using OpenCV, dlib facial landmarks, and NumPy. |
| `test2/min_distance_point.py` | Script for finding a point with the minimum total distance to a set of points using Weiszfeld's algorithm and visualizing the result. |

> Note: `face_skin_mask.py` expects a dlib landmark predictor file, for example `shape_predictor_68_face_landmarks.dat`. The model file itself is not stored in the repository, so it should be downloaded separately and passed with `--predictor`.

Example for `test1`:

```bash
python "yadro/YADRO Импульс/test1/face_skin_mask.py" input.jpg output.png --predictor shape_predictor_68_face_landmarks.dat
```

Example for `test2`:

```bash
python "yadro/YADRO Импульс/test2/min_distance_point.py"
```

### `yandex/agents_week`

Notebook with a homework assignment on AI agents. It includes a template for working with LLM calls, tools, tool tracing, and a small product-catalog environment for agent experiments.

### `yandex/vsosh`

A set of machine learning homework notebooks:

| Notebook | Topic |
|---|---|
| `vsosh_ht_1.ipynb` | Exploratory data analysis and visualization on the Netflix dataset |
| `vsosh_ht_2.ipynb` | Heart Disease classification with EDA and model comparison |
| `vsosh_ht_3.ipynb` | Regression model comparison on the Diamonds dataset |
| `vsosh_ht_4.ipynb` | Text classification / analysis of bank customer reviews |
| `vsosh_ht_5.ipynb` | Word embeddings, FastText training, and semantic search for similar questions |

### `yandex/ya_spec`

Notebook on introductory neural networks. It includes work with MNIST, PyTorch, training loops, validation, testing, and classification metrics.

## Main topics

- Exploratory data analysis
- Data visualization
- Classical machine learning
- Classification and regression
- Text processing and vectorization
- Word embeddings and semantic search
- Neural networks with PyTorch
- AI agents and tool usage
- Computer vision
- Facial landmark processing
- Geometric optimization

## Technologies

The repository uses different parts of the Python ecosystem, including:

- Python
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- scikit-learn
- PyTorch
- torchvision
- gensim
- FAISS
- LangChain
- OpenCV
- dlib

## How to run

Clone the repository:

```bash
git clone https://github.com/amemeansrain/ML-Practices.git
cd ML-Practices
```

Install the required packages depending on the task or notebook you want to run.

For the notebook-based ML tasks:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

For neural-network notebooks:

```bash
pip install torch torchvision
```

For embeddings and semantic search:

```bash
pip install gensim faiss-cpu
```

For AI agents:

```bash
pip install langchain-openai langchain-core
```

For the YADRO computer-vision task:

```bash
pip install opencv-python dlib numpy
```

Start Jupyter for notebooks:

```bash
jupyter notebook
```

Then open the needed notebook from the `yandex/` directory.

## Notes

This repository is mainly educational. The notebooks and scripts are used for practice, experimentation, and tracking progress while studying machine learning, data analysis, AI systems, and applied Python tasks.
