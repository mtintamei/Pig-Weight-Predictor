# 🐷 Predicting Pig End Weight Based on Feed Intake
**Author:** Musa Tintamei Siongo  
**Project Type:** Portfolio / Learning Showcase  
**Dataset Source:** [SCF Dataset of 100 fattening pigs](https://zenodo.org/records/6626445)  
**Tools Used:** pandas, sklearn, matplotlib, seaborn, LinearRegression

---

## 🧠 Problem Statement
Fattening pigs is a costly process. Can we predict the final weight of a pig after fattening using only information available *before* slaughter — such as initial weight before fattening period, feed intake during the fattening period, and the fattening duration? If so, farmers could better estimate:
- Profitability
- Feeding plans
- Ideal selling time

---

## ⚠️ Disclaimer
This model is a learning artifact. It is trained on experimental data and **should not be used for real-world livestock decision-making**. The purpose is to demonstrate data science thinking, modeling skills, and end-to-end process design.

## 🧪 Model

A simple **linear regression model** was trained using:

```python
Features:
  - start_weight
  - total_feed_intake_kg
  - days_in_fattening
  - feed_type (categorical)
Target:
  - end_weight
