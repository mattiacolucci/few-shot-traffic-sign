import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose
from dataset.utils import Datum  # Import the Datum class


class CustomDataset(Dataset):
    def __init__(self, root, num_shots=None, split='train', transform=None):
        """
        Custom dataset class for loading images and labels.

        :param root: Path to the dataset root directory.
        :param num_shots: Number of samples per class (for few-shot learning). If None, use all samples.
        :param split: Dataset split ('train', 'val', or 'test').
        :param transform: Transformations to apply to the images.
        """
        self.root = root  # Root directory containing train/val/test folders
        self.num_shots = num_shots
        self.transform = transform
        self.loader = default_loader

        # Load dataset metadata and samples for all splits
        self.classnames = sorted(os.listdir(os.path.join(self.root, 'train')))  # List of class names
        self.train_x = self._load_samples(split='train')
        self.val = self._load_samples(split='val')
        self.test = self._load_samples(split='test')

        # Template for class text prompts used in CLIP
        self.template = [
            # Impressionism
            "An impressionist painting by {} featuring light and color over detail.",
            "A landscape with soft brushstrokes and fleeting light, painted by {}.",
            "A vibrant outdoor scene in the impressionist style by {}.",
            "A painting that captures the mood of a moment by {}.",

            # Realism
            "A realistic depiction of everyday life painted by {}.",
            "An artwork by {} showing lifelike figures and true-to-life details.",
            "A detailed, representational painting in the style of {}.",
            "A scene from daily life, painted with photographic accuracy by {}.",

            # Romanticism
            "A dramatic, emotional painting by {} in the romantic style.",
            "A scene of nature’s grandeur or human passion by {}.",
            "A heroic or tragic figure portrayed by {} in a romantic setting.",
            "An expressive and imaginative work of art created by {}.",

            # Expressionism
            "A highly emotional and distorted painting by {}.",
            "An artwork by {} that uses color and form to convey inner feelings.",
            "An expressionist work showing raw emotion and bold brushwork by {}.",
            "A dramatic, unsettling image created by {} to reflect psychological states.",

            # Post-Impressionism
            "A vivid, structured painting in the post-impressionist style by {}.",
            "A stylized landscape with bold colors and defined shapes by {}.",
            "An emotionally charged and symbolic artwork painted by {}.",
            "A post-impressionist composition by {} that emphasizes form over light.",

            # Modern
            "A modernist artwork by {} exploring new artistic ideas.",
            "An innovative painting by {} breaking traditional conventions.",
            "An abstract or conceptual piece by {} from the modern art movement.",
            "A simplified and experimental composition by {}.",

            # Baroque
            "A dramatic and ornate Baroque painting by {}.",
            "An artwork with intense contrast and theatrical scenes by {}.",
            "A richly detailed and emotional religious scene painted by {}.",
            "A dynamic composition with grandeur and movement by {}.",

            # Surrealism
            "A surreal dream-like painting by {} filled with unexpected imagery.",
            "An artwork by {} that blends reality with the subconscious.",
            "A bizarre and symbolic scene created in the surrealist style by {}.",
            "A painting of a fantastical world rendered by {}.",

            # Symbolism
            "A symbolic and mysterious composition by {}.",
            "An artwork by {} that represents abstract ideas through imagery.",
            "A deeply emotional and metaphorical painting in the style of {}.",
            "A mythological or spiritual scene painted by {} using symbolic language.",

            # Abstract Expressionism
            "An energetic, gestural abstract painting by {}.",
            "A large-scale artwork filled with dynamic brushwork and emotion by {}.",
            "A non-representational expression of feeling in the style of {}.",
            "An intense and spontaneous composition created by {}."
        ]


        # Add train, val, and test attributes (assuming the dataset is already split)
        if split == 'train':
            self.samples = self.train_x
        elif split == 'val':
            self.samples = self.val
        elif split == 'test':
            self.samples = self.test
        pass

    def _load_samples(self, split):
        """
        Load the dataset samples and labels from the specified split directory.
        Assumes the directory structure:
        root/
            train/
                class_1/
                    img1.jpg
                    img2.jpg
                    ...
                class_2/
                    img1.jpg
                    img2.jpg
                    ...
            val/
                class_1/
                    img1.jpg
                    img2.jpg
                    ...
                class_2/
                    img1.jpg
                    img2.jpg
                    ...
            test/
                class_1/
                    img1.jpg
                    img2.jpg
                    ...
                class_2/
                    img1.jpg
                    img2.jpg
                    ...
        """
        split_dir = os.path.join(self.root, split)
        samples = []
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classnames)}
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.endswith(('.jpg', '.png'))]
            if self.num_shots:
                images = images[:self.num_shots]  # Limit to num_shots if specified
            for img in images:
                samples.append(Datum(impath=img, label=cls_idx, domain=-1, classname=cls_name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample.

        :param idx: Index of the sample.
        :return: A Datum object.
        """
        return self.samples[idx]