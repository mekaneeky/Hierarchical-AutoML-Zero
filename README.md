# Hivetrain AutoML Subnet

Hivetrain's Incentivized AutoML Loss Subnet, a collaborative platform dedicated to revolutionizing deep learning by automating the discovery of improved and novel neural network components such loss functions, activation functions and potentially new algorithms that surpass the current state-of-the art. Inspired by the AutoML Zero paper, we leverage genetic programming to progressively develop increasingly complex mathematical functions using evolutionary and gradient based optimization.

Traditionally, critical components of neural networks—such as loss functions, layers, optimizers, and activation functions—are painstakingly crafted by human researchers. However, with the remarkable advancements in AI, we are poised to leverage its capabilities to surpass human ingenuity in AI research. Our subnet aims to enable AI to autonomously innovate and excel in the creation of these essential components.

## Current Focus

Currently running on Bittensor netuid (coming soon) (100 testnet), we're starting with a loss function search where miners are incentivesed to find better loss functions for a neural networks.

The search for effective loss functions is a critical aspect of advancing deep learning. Loss functions play a pivotal role in guiding the training process of complex models, such as neural networks, by quantifying the difference between predicted outputs and actual targets. An optimal loss function can significantly enhance a model's ability to learn from large and intricate datasets, improve convergence rates, and ultimately lead to better generalization on unseen data. As deep learning applications grow increasingly sophisticated, the need for customized loss functions tailored to specific tasks—such as image classification, natural language processing, or generative modeling—becomes more pronounced.

In recent years, traditional loss functions have faced challenges in addressing unique deep learning complexities, such as handling class imbalance, noise, and varying data distributions. This is where loss function search becomes essential; it enables researchers and practitioners to automate the discovery of innovative loss functions that can outperform standard ones in deep learning contexts. By leveraging advanced techniques such as genetic algorithms and automated machine learning (AutoML), the search for new loss functions not only accelerates the model development process but also pushes the boundaries of what is achievable in deep learning. Refining loss functions can lead to more robust and accurate models, fostering advancements across various industries, from healthcare to autonomous systems, where the performance and reliability of deep learning models are paramount.

## Roadmap

Future steps include scaling up the complexity and generality of evaluations as well as expanding the search space to more AI algorithm components (losses, activations, layer types). Due to the research aligned nature of this subnet, new experiments and code updates are expected frequently and will be announced earlier on the Hivetrain discord server as well as the bittensor subnet discord channel.



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
