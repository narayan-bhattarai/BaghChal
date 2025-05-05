# 🐅🐐 BaghChal AI with Reinforcement Learning

This project implements and compares various reinforcement learning strategies for the traditional Nepali board game **BaghChal** (Tiger and Goat). It features custom environments, training procedures, and evaluation for four RL algorithms: **Q-Learning, SARSA, Expected SARSA, and Deep Q-Network (DQN)**. You can also play against the trained agents via the terminal.

## 🧠 Algorithms Implemented

- 🔵 Q-Learning
- 🟡 SARSA
- 🟢 Expected SARSA (eSARSA)
- 🔴 DQN (Deep Q-Learning with PyTorch)

Each algorithm is trained in a self-play setup, and their performance is evaluated based on:
- 📈 Reward progression over time
- 🥇 Win-rate distribution between tigers and goats
- 🧭 Exploration decay and convergence trends

---

## 🎮 Game: BaghChal (Tiger and Goat)

BaghChal is a two-player asymmetric strategy game:
- 🐅 4 Tigers try to **capture** goats by jumping over them.
- 🐐 20 Goats try to **immobilize** all tigers by blocking their moves.

📚 Learn more: [Wikipedia - BaghChal](https://en.wikipedia.org/wiki/Bagh-Chal)

---

## 🛠️ Project Structure

| File                        | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `DQN.py`                   | Full implementation of environment, agents, training, UI     |
| `goat_dqn_model.pt`        | Trained goat DQN model (PyTorch)                             |
| `tiger_dqn_model.pt`       | Trained tiger DQN model (PyTorch)                            |
| `*_q_table.pkl`            | Saved Q-tables for SARSA, Q-Learning, Expected SARSA         |
| `README.md`                | You're reading it!                                           |

---

## 🚀 How to Run

### 🔧 Install Requirements
```bash
pip install matplotlib torch numpy scikit-learn
```

### 🏋️ Train Agents (Optional)
Each algorithm has a training loop in `DQN.py`. Run it directly:
```bash
python DQN.py
```

Models and Q-tables are automatically saved after training.

---

## 📊 Visualizations

The script generates multiple useful plots:
- 📈 **Training Reward Trend**
- 📉 **Epsilon Decay (Exploration vs Exploitation)**
- 🧮 **Win Distribution per Algorithm**
- 🧠 **Moving Average Reward**
- 🔍 **Confusion Matrix (optional for DQN/SARSA)**

Example:
```
🐐 Goat won 70% games in DQN
🐅 Tiger dominated in SARSA with 80% win rate
```

You can modify and save these plots inside the code using `matplotlib`.

---

## 📋 Evaluation Summary

| Algorithm      | 🐅 Tiger Win Rate | 🐐 Goat Win Rate | ⚖️ Notes                                         |
|----------------|-------------------|------------------|-------------------------------------------------|
| SARSA          | 80%               | 20%              | Best performing for tiger agents                |
| Q-Learning     | 60%               | 40%              | Decent baseline                                 |
| Expected SARSA | 70%               | 30%              | Slightly below SARSA                            |
| DQN            | 30%               | 70%              | Goats learned strong defense via function approx|

---

## 🕹️ Human vs AI - Terminal Game

To play against trained agents:
```python
user_play()
```

- 👤 Choose role: `goat` or `tiger`
- 🎯 Select actions from printed list
- 🧩 Game board rendered with emojis:
  - ⬜ Empty
  - 🐐 Goat
  - 🐅 Tiger

---

## 🧪 Research & Paper Context

This code was developed for a term research paper comparing RL algorithms for multi-agent systems. Inspired by:
- 🤖 AlphaGo’s success in Go
- 🎲 Reinforcement learning in asymmetric environments
- 📚 Prior work on Heterogeneous Q-Learning in BaghChal

---

## 👨‍💻 Authors

- 👨‍🎓 Narayan Bhattarai  
- 👨‍🎓 Deepak Pradhan  
📧 Emails:  
`narayan.bhattarai@coyotes.usd.edu`  
`deepak.pradhan@coyotes.usd.edu`

---

## 📄 License

This repository is intended for academic and research purposes only.  
Please give credit if you use or extend this work. 🙏

---

## 🙌 Acknowledgments

- 🏛️ University of South Dakota – Dept. of Computer Science  
- 👨‍🏫 Prof. Dr. Srikant Baride (Supervision & Feedback)  
- 📖 Sutton & Barto – RL textbook  
- 🐙 OpenAI Gym & GitHub community inspirations  

---
