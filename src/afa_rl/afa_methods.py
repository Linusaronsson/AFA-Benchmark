from tensordict.nn import TensorDictModule
import torch
from tensordict import TensorDict
from torch.distributions import Categorical
from torchrl.data import TensorSpec
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, QValueActor

from afa_rl.agents import Shim2018ValueModule, Zannone2019PolicyModule
from afa_rl.models import PointNet, ReadProcessEncoder, ShimEmbedder, ShimEmbedderClassifier, ShimMLPClassifier, Zannone2019PretrainingModel
from afa_rl.utils import remove_module_prefix
from common.custom_types import AFAMethod, AFASelection, FeatureMask, Label, MaskedFeatures


def get_td_from_masked_features(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
) -> TensorDict:
    """
    Creates a TensorDict including the keys
    - "action_mask"
    - "masked_features"
    - "feature_mask"
    from the masked features and the feature mask.
    """
    # The action mask is almost the same as the negated feature mask but with one extra element
    action_mask = torch.ones(
        masked_features.shape[0],
        masked_features.shape[1] + 1,
        dtype=torch.bool,
    )
    action_mask[:, 1:] = ~feature_mask

    td = TensorDict(
        {
            "action_mask": action_mask,
            "masked_features": masked_features,
            "feature_mask": feature_mask,
        },
        batch_size=masked_features.shape[0],
    )

    return td


class Shim2018AFAMethod(AFAMethod):
    """
    Implements the AFAMethod protocol for the Shim2018 agent.
    """

    def __init__(
        self,
        device: torch.device,
        qvalue_actor: QValueActor,
        embedder_and_classifier: ShimEmbedderClassifier, # contains a reference to the embedder, even though it's already contained within the qvalue_actor
    ):
        self.device = device

        # Load models, set them to eval mode and disable gradients
        self.qvalue_actor = qvalue_actor.to(self.device)
        self.qvalue_actor.eval()
        self.qvalue_actor.requires_grad_(False)
        self.embedder_and_classifier = embedder_and_classifier.to(self.device)
        self.embedder_and_classifier.eval()
        self.embedder_and_classifier.requires_grad_(False)

    def select(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        td = get_td_from_masked_features(masked_features, feature_mask)

        # Apply the agent's policy to the tensordict
        # with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.qvalue_actor(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # with torch.no_grad():
        embedding, logits = self.embedder_and_classifier(masked_features, feature_mask)
        probs: Label = logits.softmax(dim=-1)
        return probs

    def save(self, path: str):
        torch.save(
            {
                "qvalue_actor": self.qvalue_actor.cpu(),
                "embedder_and_classifier": self.embedder_and_classifier.cpu(),
            },
            path,
        )

    @staticmethod
    def load(path: str, device: torch.device) -> "Shim2018AFAMethod":
        """
        Loads the Shim2018AFAMethod object, including its components.
        """
        data = torch.load(path, weights_only=False, map_location=device)

        qvalue_actor = data["qvalue_actor"].to(device)
        embedder_and_classifier = data["embedder_and_classifier"].to(device)

        return Shim2018AFAMethod(
            device=device,
            qvalue_actor=qvalue_actor,
            embedder_and_classifier=embedder_and_classifier,
        )


class Zannone2019AFAMethod(AFAMethod):
    """
    Implements the AFAMethod protocol for the Zannone2019 agent.
    """

    def __init__(
        self,
        device: torch.device,
        actor_network: ProbabilisticActor,
        pretrained_model: Zannone2019PretrainingModel
    ):
        self.device = device
        self.actor_network = actor_network
        self.pretrained_model = pretrained_model

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        td = get_td_from_masked_features(masked_features, feature_mask)

        # Apply the agent's policy to the tensordict
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.actor_network(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection

    def predict(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Label:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        with torch.no_grad():
            encoding, mu, logvar, z = self.pretrained_model.partial_vae.encode(masked_features, feature_mask)
            logits = self.pretrained_model.classifier(mu)

        probs: Label = logits.softmax(dim=-1)
        return probs

    def save(self, path: str):
        torch.save(
            {
                "actor_network": self.actor_network.cpu(),
                "pretrained_model": self.pretrained_model.cpu(),
            },
            path,
        )

    @staticmethod
    def load(path: str, device: torch.device) -> "Zannone2019AFAMethod":
        """
        Loads the Zannone2019AFAMethod object, including its components.
        """
        data = torch.load(path, weights_only=False, map_location=device)

        actor_network = data["actor_network"].to(device)
        pretrained_model = data["pretrained_model"].to(device)

        return Zannone2019AFAMethod(
            device=device,
            actor_network=actor_network,
            pretrained_model=pretrained_model,
        )


class RandomDummyAFAMethod(AFAMethod):
    """
    A dummy AFAMethod for testing purposes. Chooses a random feature to observe from the masked features.
    """

    def __init__(self, device: torch.device, n_classes: int):
        self.device = device
        self.n_classes = n_classes

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        """
        Chooses to observe a random feature from the masked features (or stop collecting features).
        """
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # Sample from unobserved features uniformly
        probs = (~feature_mask).float()

        # Avoid division by zero
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = torch.where(
            row_sums > 0, probs / row_sums, probs
        )  # normalize or leave zeros

        # Sample one index per row
        sampled = torch.multinomial(probs, num_samples=1)
        selection = sampled.squeeze(1)  # (B, 1) → (B,)

        return selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Returns a random prediction from the classes.
        """
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # Pick a random class from the classes
        prediction = torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )
        # One-hot encode the prediction
        prediction = torch.nn.functional.one_hot(
            prediction, num_classes=self.n_classes
        ).float()

        return prediction

    def save(self, path: str) -> None:
        """
        Saves the method to a file.
        """
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path,
        )

    @staticmethod
    def load(path: str, device: torch.device) -> "RandomDummyAFAMethod":
        """
        Loads the method from a file.
        """
        data = torch.load(path)
        return RandomDummyAFAMethod(device, data["n_classes"])


class SequentialDummyAFAMethod(AFAMethod):
    """
    A dummy AFAMethod for testing purposes. Always chooses the next feature to observe in order.
    """

    def __init__(self, device: torch.device, n_classes: int):
        self.device = device
        self.n_classes = n_classes

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # Choose the next unobserved feature
        unobserved_features = (~feature_mask).nonzero(as_tuple=True)[1]
        if unobserved_features.numel() == 0:
            return torch.tensor(0, device=masked_features.device)
        selection = unobserved_features[0] + 1

        return selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Returns a random prediction from the classes.
        """
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # Pick a random class from the classes
        prediction = torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )
        # One-hot encode the prediction
        prediction = torch.nn.functional.one_hot(
            prediction, num_classes=self.n_classes
        ).float()

        return prediction

    def save(self, path: str) -> None:
        """
        Saves the method to a file.
        """
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path,
        )

    @staticmethod
    def load(path: str, device: torch.device) -> "SequentialDummyAFAMethod":
        """
        Loads the method from a file.
        """
        data = torch.load(path)
        return SequentialDummyAFAMethod(device, data["n_classes"])
