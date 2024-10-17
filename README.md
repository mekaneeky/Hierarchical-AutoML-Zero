# Hivetrain AutoML Subnet

Hivetrain's Incentivized AutoML Loss Subnet, a collaborative platform dedicated to revolutionizing deep learning by automating the discovery of improved and novel neural network components such loss functions, activation functions and potentially new algorithms that surpass the current state-of-the art. Inspired by the AutoML Zero paper, we leverage genetic programming to progressively develop increasingly complex mathematical functions using evolutionary and gradient based optimization.

Traditionally, critical components of neural networks—such as loss functions, layers, optimizers, and activation functions—are painstakingly crafted by human researchers. However, with the remarkable advancements in AI, we are poised to leverage its capabilities to surpass human ingenuity in AI research. Our subnet aims to enable AI to autonomously innovate and excel in the creation of these essential components.

## Current Focus

Currently running on Bittensor netuid (coming soon) (100 testnet), we're starting with a loss function search where miners are incentivesed to find better loss functions for a neural networks.

## Roadmap

Future steps include scaling up the complexity and generality of evaluations as well as expanding the search space to more AI algorithm components (losses, activations, layer types). Due to the research aligned nature of this subnet, new experiments and code updates are expected frequently and will be announced earlier on the Hivetrain discord server as well as the bittensor subnet discord channel.

### Why This Is Needed
Deep learning models have achieved remarkable success across various domains, from computer vision and natural language processing to reinforcement learning and beyond. However, these models often rely on hand designed features. AI has proven superhuman performance in many domains, including chess, go, medical diagnostics and music generation. We think that AI research should be added to this list. By training AI to design traditionally hand designed features and components of AI algorithms we move towards self-improving AI and superintelligence. 

## Participation

### As a Miner

You have two main approaches as a miner:

1. **Rely on Brains:**
   - Develop new functions in the target optimization area and write algorithms in our genetic format.
   - Create better optimization approaches than our default offerings.
   - Design more efficient miners to maximize your available compute resources.

2. **Rely on Compute:**
   - If you have enough computational resources on your own:
     - Run an independent miner.
   - If you don't:
     - Joing a mining pool (work in progress)

### As a Validator

We welcome validators and are committed to supporting you. We can assist with setup, automation, cost-reduction, and other measures to reduce friction. Please note: Do not copy weights.

## FAQs

**Q: Is there research backing your claims?**  
A: Yes, our work is inspired by and based on several research papers:
- [AutoML-Zero: Evolving Machine Learning Algorithms From Scratch](https://arxiv.org/abs/2003.03384)
- [Lion: Adversarial Distillation of Closed-Source Large Language Model](https://arxiv.org/abs/2302.06675)
- For more AutoML research areas, refer to the [AutoML Conference 2024](https://2024.automl.cc/)

**Q: Are you done with distributed training?**  
A: We're still developing our distributed training solution, experiments are running. 

## Getting Started

For detailed instructions on setting up and running miners and validators, please refer to our [Miner and Validator Tutorial](docs/tutorial.md).

## Community and Support

Join our community channels for discussions, support, and updates:
- [Bittensor Discord](https://discord.com/channels/799672011265015819/1174839377659183174)
- [Hiveτrain Discord](https://discord.gg/JpRSqTBBZU)

---

We're excited to have you join us on this journey of distributed AutoML. Let's build the hive.
