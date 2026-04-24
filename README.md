# qmarg

Research prototype for one-dimensional two-center soft-Coulomb LCAO
basis comparison with an algebraic truncated Hamiltonian backend.

一次元二中心ソフトクーロンLCAO基底比較の研究プロトタイプ。
代数的打切りハミルトニアンバックエンドを含む。

---

## Backend / バックエンド

- **real_space**: exact quadrature-based H and S (reference)  
  数値積分による厳密な H, S — 参照解
- **algebraic_truncated**: exact overlap S, cutoff-dependent H via truncated Gaussian expansions.  
  重なりSは厳密、HはGaussian展開の打切り次数(cutoff)に依存。
  Eigenvalues converge oscillatorily with increasing cutoff.  
  固有値はcutoff増大とともに振動しながら収束する。

---

## Quick start / 実行方法

All experiments are manual diagnostic scripts (no CLI, no CI):  
すべて手動診断スクリプト（CLI・CI不要）。

```bash
# Cutoff convergence at fixed basis size / 固定基底でのcutoff収束
python3 results/run_truncated_backend_convergence.py

# Cutoff x basis-size grid / cutoff×基底サイズグリッド
python3 results/run_cutoff_basis_grid.py

# Extended sweep (fpc 1..8) / 拡張スイープ
python3 results/run_truncated_backend_2d_sweep.py

# Cutoff selection rule table / cutoff選択ルール表
python3 results/run_cutoff_selection_summary.py

# Stability classification / 安定性分類
python3 results/run_stability_classification.py

# Basis error vs cutoff error decomposition / 誤差分解
python3 results/run_error_decomposition.py

# Solver instability diagnostics / solver不安定性診断
python3 results/run_backend_diagnostics.py

# Odd/even cutoff parity check / 奇偶cutoffパリティ確認
python3 results/run_cutoff_odd_even_check.py
```

---

## Key findings / 主要発見

### 1. S is exact / Sは厳密
Overlap matrix error is ~2e-16 across all basis sizes and cutoffs.  
全基底サイズ・全cutoffで重なり誤差 ~2e-16。

### 2. Oscillatory eigenvalue convergence / 固有値の振動的収束
Eigenvalue errors change sign between successive cutoffs  
(e.g., E1_err: + → - → + → -).  
固有値誤差はcutoffの増加ごとに符号反転を繰り返す。

This is genuine convergence behaviour, not a parity artifact.  
偶奇のアーティファクトではない（奇数cutoffでも包絡線上に乗る）。

Use absolute error and the last-two-cutoff delta to characterize convergence robustly.  
絶対値誤差と直前cutoffとの差δEを併用して評価すること。

### 3. Cutoff selection rule / cutoff選択ルール

| Tolerance / 許容誤差 | Minimum cutoff / 最小cutoff | Reachable? / 到達？ |
|-----------------------|----------------------------|---------------------|
| loose (1e-3)          | K ≥ 10–12                 | Yes / 可            |
| medium (1e-4)         | K ≥ 16 (estimated / 推定) | No (K ≤ 12) / 不可  |
| strict (1e-5)         | K ≫ 12                    | No (basis-limited) / 不可（基底誤差限界） |

Cutoff requirement is approximately independent of basis size for fpc ≤ 4.  
fpc ≤ 4 では必要cutoffは基底サイズにほぼ依存しない。

### 4. Error decomposition / 誤差分解
At K ≥ 10, cutoff error ≪ basis error for fpc ≥ 2.  
K ≥ 10 では打切り誤差 ≪ 基底誤差。

E2 basis error ~7e-3 dominates — larger basis needed for ε < 1e-3.  
E2の基底誤差 ~7e-3 が支配的。ε < 1e-3 にはより大きな基底が必要。

### 5. Backend instability at large basis (fpc ≥ 6)  
   大基底でのバックエンド不安定性
Truncation occasionally flips the sign of near-zero H eigenvalues (λ ~ 10⁻⁷).  
打切りがHの近零固有値(λ ~ 10⁻⁷)の符号を反転させることがある。

The S⁻¹/² transformation amplifies the sign-flipped mode, producing spurious generalized eigenvalues (E1_err up to 10⁵).  
S⁻¹/²変換が符号反転モードを増幅し、spuriousな汎化固有値(E1誤差 10⁵)を生成する。

fpc ≤ 4 is always stable. / fpc ≤ 4 は常に安定。

---

## Planned Prolog integration / Prolog連携計画

Prolog is intended as a symbolic formula generator and rule checker;  
Python remains responsible for numerical evaluation, matrix assembly, eigensolvers, and convergence experiments.  
Prologは記号式生成・規則チェック、Pythonは数値評価・行列組立・固有値解法・収束実験を担当。

See [`docs/prolog_plan.md`](docs/prolog_plan.md).

---

## SU(1,1) Gaussian-operator roadmap / SU(1,1) Gaussian作用素ロードマップ

Fixes the non-unitary Gaussian multiplication operator convention before adding a second algebraic backend.  
第二の代数的バックエンド追加に先立ち、非ユニタリGaussian乗積演算子の規約を確定する。

See [`docs/su11_gaussian_operator.md`](docs/su11_gaussian_operator.md).
