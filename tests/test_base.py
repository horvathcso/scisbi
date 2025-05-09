import pytest
from scisbi.base.inference import BaseInferenceAlgorithm


# Dummy implementations for testing
class DummySimulator:
    def simulate(self, *args, **kwargs):
        return "simulated_data"


class DummyPrior:
    def log_prob(self, *args, **kwargs):
        return -1.0

    def sample(self, *args, **kwargs):
        return "sampled_parameter"


class DummySummaryStatistic:
    def compute(self, data):
        return "summary:" + str(data)


# Concrete implementation for testing
class DummyInferenceAlgorithm(BaseInferenceAlgorithm):
    def infer(self, observed_data, num_simulations, **kwargs):
        simulator_result = self.simulator.simulate()
        if self.summary_statistic is not None:
            simulated_value = self.summary_statistic.compute(simulator_result)
        else:
            simulated_value = simulator_result
        return {
            "simulated": simulated_value,
            "prior_sample": self.prior.sample(),
            "observed": observed_data,
            "num_simulations": num_simulations,
            "settings": self.settings,
        }


def test_successful_inference():
    simulator = DummySimulator()
    prior = DummyPrior()
    summary = DummySummaryStatistic()
    algo = DummyInferenceAlgorithm(simulator, prior, summary, param1="value1")
    result = algo.infer("observed_data", 100)

    assert result["simulated"] == "summary:simulated_data"
    assert result["prior_sample"] == "sampled_parameter"
    assert result["observed"] == "observed_data"
    assert result["num_simulations"] == 100
    assert result["settings"]["param1"] == "value1"


def test_inference_without_summary():
    simulator = DummySimulator()
    prior = DummyPrior()
    algo = DummyInferenceAlgorithm(simulator, prior)
    result = algo.infer("observed_data", 50)

    assert result["simulated"] == "simulated_data"
    assert result["num_simulations"] == 50


def test_invalid_simulator():
    class BadSimulator:
        pass  # Does not implement simulate

    prior = DummyPrior()
    with pytest.raises(
        TypeError,
        match="simulator must be an instance of a class with a 'simulate' method",
    ):
        DummyInferenceAlgorithm(BadSimulator(), prior)


def test_invalid_prior_missing_log_prob():
    simulator = DummySimulator()

    class BadPrior:
        def sample(self):
            return "sampled_value"

    with pytest.raises(
        TypeError, match="prior must be an instance of a class with a 'log_prob' method"
    ):
        DummyInferenceAlgorithm(simulator, BadPrior())


def test_invalid_prior_missing_sample():
    simulator = DummySimulator()

    class BadPrior:
        def log_prob(self):
            return -1.0

    with pytest.raises(
        TypeError, match="prior must be an instance of a class with a 'sample' method"
    ):
        DummyInferenceAlgorithm(simulator, BadPrior())


def test_invalid_summary_statistic():
    simulator = DummySimulator()
    prior = DummyPrior()

    class BadSummary:
        pass  # Does not implement compute

    with pytest.raises(
        TypeError,
        match="summary_statistic, if provided, must be an instance of a class with a 'compute' method",
    ):
        DummyInferenceAlgorithm(simulator, prior, BadSummary())
