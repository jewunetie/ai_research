# Labor-for-Content-Access: Data Labeling as Alternative Content Monetization

Research prototype exploring a novel content monetization model where users choose between viewing ads, paying subscriptions, or performing data labeling tasks to access creator content.

## Overview

This is **NOT** a CAPTCHA or bot detection system. It's an explicit, transparent user choice mechanism for alternative content monetization that:
- Gives users agency in how they "pay" for content
- Provides creators with an alternative to advertising revenue
- Generates high-quality labeled data for ML companies
- Explores data dignity and fair compensation models

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

### Running the Simulation

```bash
# Run with default configuration
python src/main.py

# Run with custom configuration
python src/main.py --config config/experiments/baseline.yaml
```

## Project Structure

```
labor_content_access_model/
├── config/              # Configuration files
│   └── default.yaml     # Default simulation parameters
├── src/
│   ├── data/            # Data schemas and dataset management
│   ├── agents/          # Mesa agent implementations
│   ├── marketplace/     # Task routing and inventory
│   ├── quality/         # Quality control and consensus
│   ├── revenue/         # Revenue calculation and distribution
│   ├── analytics/       # Metrics and visualization
│   ├── simulation/      # Main simulation model
│   ├── config/          # Configuration loading
│   └── main.py          # Entry point
├── tests/               # Unit tests
├── results/             # Generated simulation results
└── IMPLEMENTATION.md    # Detailed technical design
```

## Core Research Questions

1. **Will users choose data labeling over ads/payment?**
   - Target: >10% of users choose labor option

2. **Can label revenue match advertising CPM?**
   - Target: Labor CPM within 50% of ad CPM ($1.50-$12)

3. **Does access-motivated labeling produce quality comparable to paid crowdsourcing?**
   - Target: Fleiss' kappa >0.70
   - Target: Accuracy >75% vs. ground truth

4. **What task types and UX maximize acceptance and quality?**
   - Compare task difficulty tolerance
   - Optimize task count (5-10 tasks)

5. **What revenue split feels fair?**
   - Model different splits (platform/creator/user)
   - Default: 20%/70%/10%

## Datasets Used

- **MNIST**: Handwritten digit classification (70K images, 12 MB)
- **CIFAR-10**: Object recognition in color images (60K images, 163 MB)
- **SST-2**: Sentiment analysis (11,855 sentences, 7 MB)

Total Phase 1: 182 MB

## Configuration

Edit `config/default.yaml` to customize:

- **Simulation parameters**: Steps, random seed, output directory
- **Agent populations**: Number and distribution of users, creators, companies
- **Economic parameters**: Pricing, CPM benchmarks, revenue splits
- **Quality requirements**: Minimum kappa, redundancy factor

## Key Components

### 1. Agent-Based Simulation (Mesa)
- **UserAgent**: Makes access decisions using utility functions
- **CreatorAgent**: Tracks traffic and revenue
- **CompanyAgent**: Generates labeling tasks

### 2. Marketplace Engine
- Routes tasks to users based on difficulty tolerance
- Manages task inventory and assignment

### 3. Quality Control
- Majority vote consensus
- Fleiss' kappa calculation
- Quality-based payment

### 4. Revenue Engine
- Distributes revenue: 20% platform, 70% creator, 10% user (configurable)
- Calculates CPM equivalents
- Compares labor revenue to ad revenue

## Output

After running a simulation, results are saved to `results/run_TIMESTAMP/`:

- `summary_report.json`: Comprehensive metrics
- `sessions.csv`: All content access sessions
- `simulation_results.png`: Visualization plots
- `config.yaml`: Configuration used for this run

## Example Results

```
SIMULATION SUMMARY
==================================================
## Choice Distribution
  Labor:     1543 (15.4%)
  Ads:       7234 (72.3%)
  Payment:   1223 (12.2%)

## Revenue
  Total Revenue:    $45,678.32
  Labor Revenue:    $12,345.67 (27.0%)
  Ads Revenue:      $28,901.23 (63.3%)
  Payment Revenue:  $4,431.42 (9.7%)

## Quality Metrics
  Avg Fleiss' Kappa: 0.742
  Avg Accuracy:      83.2%
  Tasks w/ Consensus: 514
  Labels Processed:  4,629
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Documentation

- `CLAUDE.md`: Project overview and scope
- `RESEARCH.md`: Literature review and prior work
- `DATA_AVAILABILITY.md`: Dataset analysis
- `IMPLEMENTATION.md`: Technical design and architecture

## Research Background

This project explores the intersection of:
- **Data dignity**: User compensation for data labor (Jaron Lanier)
- **Platform cooperatives**: Fair revenue sharing
- **Attention economy**: Alternatives to advertising
- **Crowdsourcing quality**: Consensus algorithms and quality control
- **Creator economy**: Sustainable monetization models

## License

Research prototype for academic/educational use.

## Contact

For questions about this research prototype, please refer to the documentation or open an issue.
