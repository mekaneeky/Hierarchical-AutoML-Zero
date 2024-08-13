class FitnessEvaluator:
    def __init__(self, desired_outputs):
        self.desired_outputs = desired_outputs

    def evaluate(self, genome, input_data):
        output = genome.execute(input_data)
        #if isinstance(output, torch.Tensor):
        #    output = output.mean().item()  # Convert tensor to scalar
        #if isinstance(self.desired_outputs, torch.Tensor):
        #    desired_output = self.desired_outputs.mean().item()
        #else:
        desired_output = self.desired_outputs
        try:
            return 1.0 / (1.0 + abs(output - desired_output).sum())
        except:
            return 0.0