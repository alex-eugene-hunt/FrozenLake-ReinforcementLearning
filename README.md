# FrozenLake Reinforcement Learning Project

## Project Overview
This project implements and compares different reinforcement learning algorithms on the FrozenLake environment from OpenAI Gymnasium. The implementation focuses on analyzing the effectiveness of various RL techniques, including Temporal Difference learning with heuristics and SARSA(λ) with eligibility traces.

## Technical Implementation

### Reinforcement Learning Algorithms

1. **Temporal Difference (TD) Learning**
   - Custom implementation with heuristic adjustments
   - Features:
     - Q-learning algorithm implementation
     - Heuristic-based policy enhancement
     - Action-value matrix optimization
     - Opposite action mapping for improved exploration
   - Key Parameters:
     - Learning rate (α = 0.1)
     - Discount factor (γ = 0.9)
     - Episode count: 2000

2. **SARSA(λ) with Eligibility Traces**
   - Implementation with hyperparameter optimization
   - Features:
     - Eligibility traces for credit assignment
     - Epsilon-greedy action selection
     - Comprehensive hyperparameter search
   - Hyperparameter Ranges:
     - Discount factors (γ): [0.8, 0.9, 0.95]
     - Lambda values (λ): [0.4, 0.6, 0.8]
     - Learning rates (α): [0.1, 0.05, 0.01]
     - Epsilon values (ε): [0.1, 0.2, 0.05]

### Environment Configuration
- **OpenAI Gymnasium**: FrozenLake-v1
- **Custom Map Implementation**:
  - 9x9 grid world
  - States: Start (S), Frozen (F), Hole (H), Goal (G)
  - Non-slippery environment (deterministic)
  - Custom state-space design

### Technical Stack
- **Programming Language**: Python
- **Key Libraries**:
  - `gymnasium`: Environment simulation
  - `numpy`: Numerical computations
  - `matplotlib`: Performance visualization

### Performance Analysis
- Reward tracking systems:
  - Episode rewards
  - Cumulative rewards
  - Running averages
- Comparative analysis:
  - TD with/without heuristics
  - SARSA(λ) parameter optimization
- Optimal path visualization
- Policy evaluation metrics

### Code Structure
```
project/
├── AlexHunt_TemporalDifference.py  # TD Learning implementation
└── AlexHunt_EligibilityTraces.py   # SARSA(λ) implementation
```

## Technical Highlights
- Custom implementation of multiple RL algorithms
- Sophisticated hyperparameter optimization
- Advanced policy visualization
- Comparative performance analysis
- Environment customization

## Key Features
1. **Heuristic Enhancement**
   - Custom action-selection strategies
   - Opposite action mapping
   - Policy improvement techniques

2. **Hyperparameter Optimization**
   - Systematic parameter search
   - Performance comparison framework
   - Best/worst configuration analysis

3. **Visualization Tools**
   - Learning curve plotting
   - Optimal path visualization
   - Performance metrics tracking

## Skills Demonstrated
- Reinforcement Learning algorithm implementation
- Hyperparameter optimization
- Environment simulation
- Policy visualization
- Performance analysis
- Python programming
- Scientific computing

## Future Enhancements
- Implementation of additional RL algorithms (DQN, Actor-Critic)
- Enhanced exploration strategies
- Multi-environment testing
- Advanced visualization tools
- Policy transfer learning
