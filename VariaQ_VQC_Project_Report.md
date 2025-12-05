# Quantum Machine Learning using Variational Quantum Circuits

*Purna Sainath Reddy V*
School of Computer Engineering, Data Science
Manipal Institute of Technology, Bengaluru, India


---

**Abstract** — Classical machine learning faces fundamental computational barriers in high-dimensional optimization and pattern recognition at scale. Quantum Machine Learning (VariaQ) leverages quantum mechanical phenomena — superposition, entanglement, and quantum interference — to construct learning models with potential computational advantages. This project investigates Variational Quantum Circuits (VQCs), also known as Parameterized Quantum Circuits (PQCs), as a hybrid quantum-classical framework for classification and regression tasks. A synthetic dataset of 500 entries with four numerical features, encoding both binary and multi-class targets, was generated to train and evaluate fifteen quantum and classical models including VQC classifiers with angle encoding, Quantum Kernel Support Vector Machines, Hybrid Quantum-Classical Neural Networks, and entanglement-enhanced classifiers. The parameter-shift rule, quantum state formalism, and variational optimization are mathematically derived and applied. Models were assessed using Accuracy, F1 Score, ROC-AUC, R² Score, MSE, and RMSE. Results demonstrate that quantum kernel methods and entanglement-enhanced VQCs achieve competitive and often superior performance over classical counterparts on structured low-dimensional datasets, validating the mathematical foundations of VQC-based learning.

**Keywords** — Quantum Machine Learning, Variational Quantum Circuits, Parameterized Quantum Circuits, Quantum Kernel Methods, Hybrid Quantum-Classical, NISQ Devices, Parameter-Shift Rule

---

## I. INTRODUCTION

The exponential growth of data in scientific, financial, and engineering domains has intensified the demand for more powerful learning algorithms. Classical machine learning models, while highly effective, face computational bottlenecks rooted in classical physics — polynomial or exponential scaling for combinatorial optimization, kernel computation, and high-dimensional state space traversal [1], [2]. Quantum computing, grounded in the principles of quantum mechanics, offers a fundamentally different model of computation: qubits can exist in superpositions of states, quantum systems can exhibit entanglement, and interference can be exploited to amplify correct solutions and suppress incorrect ones [3].

Quantum Machine Learning (VariaQ) is an emerging interdisciplinary field at the boundary of quantum information science and machine learning that seeks to leverage quantum computational advantages for learning tasks [4]. Among the most promising near-term VariaQ approaches are Variational Quantum Circuits (VQCs) — hybrid models that use a parameterized quantum circuit evaluated on quantum hardware and optimized using classical gradient-based methods. VQCs are especially suited to Noisy Intermediate-Scale Quantum (NISQ) devices, which have limited qubit counts and coherence times but are available today [5], [6].

The mathematical foundations of VQCs span multiple areas of Mathematics for Machine Learning: linear algebra (Hilbert spaces, unitary matrices, tensor products), probability theory (Born rule, expectation values), multivariable calculus (the parameter-shift rule for quantum gradients), and optimization theory (variational methods in hybrid quantum-classical loops). This project implements, evaluates, and mathematically analyzes VQC-based learning models to demonstrate how quantum circuits encode, transform, and extract information from data.

### A. Objectives

The primary objectives of this project are:

- Implement VQC binary and multi-class classifiers using angle encoding and amplitude encoding.
- Mathematically derive the parameter-shift rule for computing quantum gradients.
- Apply quantum kernel methods and compare kernel matrices with classical RBF kernels.
- Design and evaluate a hybrid quantum-classical neural network for regression.
- Implement entanglement-enhanced VQC classifiers and analyze the role of entanglement in expressibility.
- Compare all quantum models against classical ML baselines using standard metrics.

---

## II. MATHEMATICAL FOUNDATIONS AND SYSTEM ARCHITECTURE

The VariaQ pipeline integrates quantum circuit evaluation with classical optimization. Data is classically preprocessed, encoded into quantum states, processed through parameterized unitary transformations, measured to extract expectation values, and fed into a classical optimizer that updates circuit parameters [5], [6].

**A. Qubit Formalism**

A qubit is the fundamental unit of quantum information, described by a unit vector in a two-dimensional complex Hilbert space C²:

    |ψ⟩ = α|0⟩ + β|1⟩,   α, β ∈ C,   |α|² + |β|² = 1

where |0⟩ = [1, 0]ᵀ and |1⟩ = [0, 1]ᵀ are the computational basis states. An n-qubit system lives in a 2ⁿ-dimensional Hilbert space H = C^(2ⁿ), described by tensor products: |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩.

**B. Quantum Gates and Parameterized Rotations**

Quantum gates are unitary matrices U (UU† = I) acting on qubit states. Key gates include the Hadamard H = (1/√2)[[1,1],[1,-1]], Pauli gates X, Y, Z, and CNOT for entanglement. Parameterized rotation gates used in VQCs are defined as:

    Rₓ(θ) = exp(-iθX/2) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    R_y(θ) = exp(-iθY/2) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    R_z(θ) = exp(-iθZ/2) = [[exp(-iθ/2), 0], [0, exp(iθ/2)]]

**C. Variational Quantum Circuit**

A VQC with parameter vector θ ∈ Rᵐ acts on an input-encoded state |φ(x)⟩ as:

    |ψ(x, θ)⟩ = U(θ)|φ(x)⟩ = U_L(θ_L)...U₂(θ₂)U₁(θ₁)|φ(x)⟩

The prediction is obtained via the expectation value of an observable (e.g., Pauli-Z):

    f(x, θ) = ⟨ψ(x, θ)|Z⊗I⊗...⊗I|ψ(x, θ)⟩

**D. Parameter-Shift Rule**

Classical automatic differentiation cannot be applied directly to quantum circuits. The parameter-shift rule provides an exact analytical gradient:

    ∂C/∂θⱼ = [C(θ + π/2·eⱼ) - C(θ - π/2·eⱼ)] / 2

where eⱼ is the j-th standard basis vector. This enables standard classical optimizers (Adam, SGD) to train VQCs.

**E. Quantum Kernel Function**

For quantum kernel SVM, the kernel function is defined as the inner product of quantum feature maps:

    K(x, x') = |⟨φ(x)|φ(x')⟩|²

where |φ(x)⟩ = U_enc(x)|0⟩ is the quantum feature map state. This implicitly maps data into exponentially high-dimensional Hilbert spaces.

**F. System Pipeline**

- **Data Preprocessing**: Classical normalization and encoding to [0, π] for angle encoding.
- **State Preparation**: Data encoded into qubit states via parameterized encoding layers.
- **VQC Execution**: Parameterized unitary circuit applied to encoded states.
- **Measurement**: Expectation values extracted via quantum measurement (Born rule).
- **Classical Optimization**: Cost function minimized using Adam optimizer via parameter-shift gradients.
- **Prediction**: Final measurement outcomes interpreted as class labels or regression outputs.

---

## III. DATASET

### A. Dataset Description

Since no public real-world quantum ML benchmark of appropriate dimensionality was available under privacy and platform constraints, a synthetic dataset of 500 entries was generated to simulate structured classification and regression scenarios suitable for 4-qubit quantum circuits [9]. The dataset comprises 4 numerical features and 5 target variables:

**Features (4 qubits, angle-encoded):**
- `feature_1`: Continuous, U[0, 1] — primary discriminating feature.
- `feature_2`: Continuous, U[0, 1] — secondary interaction feature.
- `feature_3`: Continuous, U[0, 1] — noise-correlated auxiliary feature.
- `feature_4`: Continuous, U[0, 1] — entanglement-sensitive feature.

**Target Variables:**
- `binary_label` (classification): Binary class (0 or 1), based on nonlinear boundary.
- `multi_label` (classification): Three-class label (0, 1, 2) based on Gaussian clusters.
- `kernel_score` (classification): Binary, based on quantum kernel separability.
- `quantum_property` (regression): Continuous target simulating a quantum expectation value.
- `entangle_class` (classification): Binary class based on entangled feature interactions.

All features are normalized to [0, 1] with no missing values. Class balance is approximately 50/50 for binary tasks and 33/33/34 for the three-class task.

*Fig. 1. Dataset overview: df.head(), df.describe(), df.info()*

### B. Data Preprocessing

Preprocessing steps included synthetic dataset generation using NumPy and Pandas, nonlinear target variable construction for each VariaQ task, feature normalization to [0, π] for angle encoding using MinMaxScaler, and an 80/20 stratified train-test split using Scikit-learn. For amplitude encoding, features were normalized to unit vectors satisfying ∑|xᵢ|² = 1. Classical baseline models used StandardScaler normalization.

---

## IV. EXPLORATORY DATA ANALYSIS

Exploratory Data Analysis (EDA) was performed to understand data distributions, inter-feature correlations, and target-feature relationships using Matplotlib and Seaborn [10]. Statistical summaries, histograms, a correlation matrix, and pairwise scatter plots were generated to identify patterns relevant to quantum encoding strategies.

Key observations:
- `feature_1` and `feature_2` exhibit a weak positive correlation (r ≈ 0.18), making them suitable for independent angle encoding per qubit.
- `feature_3` and `feature_4` are weakly anticorrelated (r ≈ −0.12), capturing complementary information that benefits entangled circuit layers.
- Binary label boundaries are nonlinear in the feature_1–feature_2 plane, motivating multi-layer VQC architectures over linear classifiers.
- The `quantum_property` regression target correlates most strongly with `feature_2` (r ≈ 0.57), suggesting that quantum expectation values are driven by single-qubit rotations on that axis.
- Class distributions across all targets are well-balanced, reducing the need for resampling techniques.

*Fig. 2. Feature distribution histograms for all dataset variables*

*Fig. 3. Correlation matrix showing pairwise feature relationships*

---

## V. METHODOLOGY

The methodology comprised five steps: (1) dataset generation and quantum preprocessing, (2) stratified train-test splitting with 80/20 ratio, (3) model construction and training across quantum and classical algorithms, (4) performance evaluation using standard metrics, and (5) model comparison and selection per use case. All VQC models were implemented using PennyLane [5] with NumPy backend, and classical models were implemented using Scikit-learn [6].

### A. Case 1: Binary Classification with VQC (Angle Encoding)

**Objective:** Predict the binary class label using a parameterized quantum circuit with angle encoding.

**Encoding:** Each feature xᵢ is encoded as a rotation gate: Rᵧ(xᵢ)|0⟩, applying four single-qubit rotations for 4 features.

**Input features:** `feature_1`, `feature_2`, `feature_3`, `feature_4`.

**Models:** VQC-1L (1 variational layer), VQC-3L (3 variational layers), Logistic Regression (classical baseline).

**Circuit Design (VQC-3L):** Angle encoding layer → [Rᵧ(θ) gates + CNOT entangling layer] × 3 → Measurement of ⟨Z⊗I⊗I⊗I⟩.

**Optimization:** Adam optimizer, learning rate = 0.01, 100 epochs, cost function = binary cross-entropy on expectation values.

**Metrics:** Accuracy, F1 Score, ROC-AUC.

**Result:** VQC-3L achieved the best classification performance (Accuracy = 0.912, F1 = 0.908, AUC = 0.965), outperforming both VQC-1L (Accuracy = 0.840) and Logistic Regression (Accuracy = 0.875). Additional variational layers increase the expressibility of the circuit, enabling it to capture the nonlinear decision boundary.

*Fig. 4. Case 1 — Confusion matrices, F1, Accuracy, and ROC curves (VQC-1L, VQC-3L, Logistic Regression)*

### B. Case 2: Multi-class Classification with VQC

**Objective:** Predict the three-class label using a multi-output VQC with amplitude encoding.

**Encoding:** Amplitude encoding normalizes the feature vector x ∈ R⁴ to ‖x‖ = 1 and encodes it as the amplitude vector of a 2-qubit state: |φ(x)⟩ = ∑xᵢ|i⟩.

**Input features:** All four features (amplitude-encoded into 2 qubits).

**Models:** VQC-Multiclass (one-vs-rest VQC), Random Forest (classical), Classical SVM with RBF kernel.

**Circuit Design:** Amplitude encoding via QRAM-inspired initialization → 2 variational layers → 3 measurement outcomes mapped to classes via softmax.

**Metrics:** Accuracy, F1 Score (macro-average), ROC-AUC (one-vs-rest).

**Result:** Random Forest achieved the highest accuracy (Accuracy = 0.883, F1 = 0.879, AUC = 0.951), outperforming VQC-Multiclass (Accuracy = 0.847, F1 = 0.839). Amplitude encoding on 2 qubits limits circuit expressibility for three-class separation, indicating that larger qubit registers would benefit multi-class VariaQ tasks.

*Fig. 5. Case 2 — F1, Accuracy, and ROC comparison (VQC-Multiclass, Random Forest, SVM-RBF)*

### C. Case 3: Quantum Kernel Support Vector Machine

**Objective:** Classify the `kernel_score` binary target using quantum kernel methods, which leverage quantum feature maps to implicitly compute inner products in exponentially large Hilbert spaces.

**Quantum Feature Map:** Each input x is encoded using the ZZ-feature map [7]:
    U_Φ(x) = exp(i∑_{j<k} φ_{jk}(x) Z_j Z_k) · ∏_j exp(i·xⱼ·Zⱼ)

**Kernel Computation:** K(xᵢ, xⱼ) = |⟨0|U_Φ†(xᵢ) U_Φ(xⱼ)|0⟩|²

**Input features:** `feature_1`, `feature_2`, `feature_3`, `feature_4`.

**Models:** QK-SVM (Quantum Kernel SVM), Classical RBF-SVM, Linear SVM.

**Metrics:** Accuracy, F1 Score, ROC-AUC.

**Result:** QK-SVM achieved the best performance (Accuracy = 0.934, F1 = 0.931, AUC = 0.978), outperforming Classical RBF-SVM (Accuracy = 0.901) and Linear SVM (Accuracy = 0.872). The quantum feature map's implicit access to a higher-dimensional kernel space enables superior separation of the quantum-defined decision boundary in `kernel_score`.

*Fig. 6. Case 3 — Quantum kernel matrix visualization, confusion matrices, F1, Accuracy, and ROC curves (QK-SVM, RBF-SVM, Linear SVM)*

### D. Case 4: Hybrid Quantum-Classical Neural Network (Regression)

**Objective:** Predict the continuous `quantum_property` target — simulating a quantum expectation value — using a hybrid model that combines classical preprocessing layers with a VQC core.

**Architecture:** Classical dense layer (4→4, ReLU) → VQC layer (4 qubits, 2 variational layers, angle encoding) → Classical output layer (1→1, linear). Gradients flow through the VQC via the parameter-shift rule.

**Input features:** `feature_1`, `feature_2`, `feature_3`, `feature_4`.

**Models:** Hybrid-QNN (VQC core), Classical MLP (3 hidden layers), Ridge Regression.

**Optimization:** Adam optimizer, learning rate = 0.005, 150 epochs, MSE loss.

**Metrics:** R² Score, MSE, RMSE.

**Result:** Hybrid-QNN achieved the best regression performance (R² = 0.821, MSE = 0.023, RMSE = 0.152), outperforming Classical MLP (R² = 0.798) and Ridge Regression (R² = 0.743). The VQC core introduces nonlinear feature transformations through quantum rotations and entanglement that are not captured by classical linear layers alone.

*Fig. 7. Case 4 — MSE, R² Score, and RMSE comparison (Hybrid-QNN, Classical MLP, Ridge Regression)*

### E. Case 5: Entanglement-Enhanced Binary Classifier

**Objective:** Predict the `entangle_class` binary target, which is defined by an interaction between `feature_3` and `feature_4`, capturing correlations that require entangled qubit operations to model efficiently.

**Entanglement Strategy:** After angle encoding, a layer of CNOT gates entangles all adjacent qubit pairs (0→1, 1→2, 2→3, 3→0), followed by parameterized Rᵧ gates, repeated for 3 layers. The entangling structure exploits quantum correlations to distinguish classes based on joint feature distributions.

**Input features:** All four features, with emphasis on feature_3–feature_4 interaction.

**Models:** VQC-Entangled (deep entangled circuit), Classical MLP, Naive Bayes.

**Metrics:** Accuracy, F1 Score, ROC-AUC.

**Result:** VQC-Entangled achieved the best performance (Accuracy = 0.896, F1 = 0.891, AUC = 0.958), outperforming Classical MLP (Accuracy = 0.878, AUC = 0.941) and Naive Bayes (Accuracy = 0.823, AUC = 0.897). Entanglement allows the circuit to capture the joint feature_3–feature_4 dependency directly at the quantum level, demonstrating a genuine structural advantage over independently processing features classically.

*Fig. 8. Case 5 — Confusion matrices, F1, Accuracy, and ROC curves (VQC-Entangled, Classical MLP, Naive Bayes)*

---

## VI. RESULTS

All VQC models were implemented in Python using PennyLane, Scikit-learn, NumPy, Pandas, Matplotlib, and Seaborn in a Jupyter Notebook environment. Classical optimizers (Adam) were applied via the parameter-shift rule for quantum gradient computation. The table below summarizes the best-model results per use case:

- **Binary VQC (Angle Encoding):** VQC-3L — Accuracy 0.912, F1 0.908, AUC 0.965.
- **Multi-class Classification:** Random Forest — Accuracy 0.883, F1 0.879, AUC 0.951.
- **Quantum Kernel SVM:** QK-SVM — Accuracy 0.934, F1 0.931, AUC 0.978.
- **Hybrid QNN Regression:** Hybrid-QNN — R² 0.821, RMSE 0.152, MSE 0.023.
- **Entanglement-Enhanced Classifier:** VQC-Entangled — Accuracy 0.896, F1 0.891, AUC 0.958.

These results confirm that quantum kernel methods excel at binary classification tasks where the decision boundary aligns with quantum feature map structure. Entanglement-enhanced VQCs outperform classical models when target labels are defined by joint feature interactions. For multi-class tasks with limited qubit counts, classical ensemble methods remain competitive. Hybrid quantum-classical architectures demonstrate advantage in regression by combining classical representation learning with quantum nonlinear transformations.

---

## VII. MODEL COMPARISON

**TABLE I. Model Strengths and Weaknesses**

| Model | Strengths | Weaknesses |
|---|---|---|
| VQC-1L | Shallow, fast training | Limited expressibility |
| VQC-3L | High expressibility, nonlinear | Many parameters, slower |
| VQC-Multiclass | Handles multi-class natively | Needs large qubit register |
| QK-SVM | Implicit high-dim kernel | Kernel matrix O(n²) compute |
| Hybrid-QNN | Classical + quantum synergy | Complex gradient flow |
| VQC-Entangled | Captures joint correlations | Sensitive to entanglement depth |
| Logistic Regression | Fast, interpretable | Linear boundary only |
| Random Forest | Robust, nonlinear | No quantum advantage |
| Classical SVM-RBF | Strong kernel baseline | Slow on large datasets |
| Classical MLP | Flexible, scalable | No quantum correlations |
| Ridge Regression | Stable, prevents overfitting | Linear only |
| Naive Bayes | Fast, probabilistic | Assumes feature independence |

---

## VIII. STRENGTHS AND LIMITATIONS

### A. Strengths

- Rigorous mathematical derivation of quantum circuit operations, encoding strategies, and the parameter-shift rule.
- Comprehensive coverage of four distinct VariaQ paradigms: angle encoding VQC, amplitude encoding VQC, quantum kernel methods, and hybrid quantum-classical networks.
- Multiple evaluation metrics used for robust and unbiased model comparison across both classification and regression tasks.
- Demonstration of genuine quantum structural advantages in Case 3 (kernel) and Case 5 (entanglement), grounded in quantum mechanics formalism.
- Hybrid architecture in Case 4 validates the practical deployment path for VariaQ on NISQ-era hardware.

### B. Limitations

- Synthetic dataset limits real-world generalizability; genuine quantum advantage remains hardware-dependent.
- Simulation of quantum circuits on classical hardware (PennyLane NumPy backend) does not capture actual quantum speedup.
- VQC training is susceptible to barren plateau phenomena — exponentially vanishing gradients in deep circuits — which limits scalability.
- Amplitude encoding requires O(2ⁿ) classical preprocessing, potentially negating quantum advantage for large feature spaces.
- Quantum kernel matrix computation scales as O(n²) in the number of training samples, which is prohibitive for large datasets.

---

## IX. CONCLUSION

This project successfully implemented and evaluated Quantum Machine Learning models based on Variational Quantum Circuits for classification and regression tasks, grounding each approach in the mathematical formalism of quantum mechanics and machine learning. Five VariaQ paradigms — VQC binary classifiers with angle encoding, multi-class VQCs with amplitude encoding, Quantum Kernel SVMs, Hybrid Quantum-Classical Neural Networks, and entanglement-enhanced VQC classifiers — were trained and assessed on a synthetic 500-entry dataset across fifteen model configurations.

Experimental results demonstrate that Quantum Kernel SVMs achieve the highest classification accuracy (0.934) by leveraging implicit exponentially high-dimensional feature maps. Entanglement-enhanced VQCs outperform classical models on targets defined by joint feature dependencies, directly validating the role of quantum entanglement in learning. Hybrid QNNs deliver superior regression performance by combining classical representational power with quantum nonlinear transformations via the parameter-shift rule. The mathematical foundations — Hilbert space formalism, unitary gates, tensor products, expectation values, and variational optimization — provide a rigorous and complete framework for understanding and extending VariaQ models beyond classical paradigms.

---

## X. FUTURE WORK

- Deploy VQC models on real IBM Quantum or IonQ hardware via Qiskit to measure actual quantum vs. simulation performance differences.
- Explore noise mitigation strategies (zero-noise extrapolation, probabilistic error cancellation) to improve NISQ-device performance.
- Address barren plateau phenomena using layer-wise training, identity block initialization, or quantum natural gradient methods.
- Apply VariaQ to real-world datasets (financial fraud, molecular property prediction) where quantum feature maps may provide genuine advantage.
- Implement QGAN (Quantum Generative Adversarial Network) architectures for synthetic quantum data generation.
- Investigate quantum transfer learning by fixing pre-trained classical layers and fine-tuning only the VQC core.
- Apply quantum-classical federated learning to enable privacy-preserving distributed VariaQ.
- Extend the mathematical analysis to include quantum Fisher information, expressibility measures, and entanglement entropy of VQC architectures.

---

## REFERENCES

[1] J. Biamonte et al., "Quantum Machine Learning," *Nature*, vol. 549, no. 7671, pp. 195–202, Sep. 2017.

[2] P. Rebentrost, M. Mohseni, and S. Lloyd, "Quantum Support Vector Machine for Big Data Classification," *Phys. Rev. Lett.*, vol. 113, no. 13, p. 130503, Sep. 2014.

[3] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, 10th anniversary ed. Cambridge University Press, 2010.

[4] M. Schuld and F. Petruccione, *Machine Learning with Quantum Computers*, 2nd ed. Springer, 2021.

[5] V. Bergholm et al., "PennyLane: Automatic Differentiation of Hybrid Quantum-Classical Computations," *arXiv preprint* arXiv:1811.04968, 2022.

[6] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[7] V. Havlíček et al., "Supervised Learning with Quantum-Enhanced Feature Spaces," *Nature*, vol. 567, no. 7747, pp. 209–212, Mar. 2019.

[8] M. Cerezo et al., "Variational Quantum Algorithms," *Nature Reviews Physics*, vol. 3, no. 9, pp. 625–644, 2021.

[9] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, "Quantum Circuit Learning," *Phys. Rev. A*, vol. 98, no. 3, p. 032309, Sep. 2018.

[10] J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007.

[11] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, "Evaluating Analytic Gradients on Quantum Hardware," *Phys. Rev. A*, vol. 99, no. 3, p. 032331, Mar. 2019.

[12] E. Farhi and H. Neven, "Classification with Quantum Neural Networks on Near Term Processors," *arXiv preprint* arXiv:1802.06002, 2018.

[13] S. Lloyd, M. Mohseni, and P. Rebentrost, "Quantum Principal Component Analysis," *Nature Physics*, vol. 10, no. 9, pp. 631–633, 2014.

[14] M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, "Parameterized Quantum Circuits as Machine Learning Models," *Quantum Sci. Technol.*, vol. 4, no. 4, p. 043001, 2019.

[15] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren Plateaus in Quantum Neural Network Training Landscapes," *Nature Communications*, vol. 9, no. 1, p. 4812, 2018.

[16] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, "Data Re-uploading for a Universal Quantum Classifier," *Quantum*, vol. 4, p. 226, 2020.

[17] T. Hubregtsen et al., "Training Quantum Embedding Kernels on Near-Term Quantum Computers," *Phys. Rev. A*, vol. 106, no. 4, p. 042431, 2022.

[18] K. Bharti et al., "Noisy Intermediate-Scale Quantum Algorithms," *Reviews of Modern Physics*, vol. 94, no. 1, p. 015004, 2022.

[19] M. Schuld, R. Sweke, and J. J. Meyer, "Effect of Data Encoding on the Expressive Power of Variational Quantum-Machine-Learning Models," *Phys. Rev. A*, vol. 103, no. 3, p. 032430, 2021.

[20] A. Abbas et al., "The Power of Quantum Neural Networks," *Nature Computational Science*, vol. 1, no. 6, pp. 403–409, 2021.
